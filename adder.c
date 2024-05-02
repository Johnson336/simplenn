#include <float.h>
#define NN_IMPLEMENTATION
#include <stdio.h>
#include "nn.h"
#include <time.h>

#include "/usr/local/include/raylib.h"

#define BITS 4

float width = 1600;
float height = 1200;

typedef struct {
  size_t *items;
  size_t count;
  size_t capacity;
} Arch;

typedef struct {
  float *items;
  size_t count;
  size_t capacity;
} Cost_Plot;

Color mixColors(Color c1, Color c2, float amt) {
  return (Color) {
    ((c1.r * amt) + (c2.r * (1-amt))),
    ((c1.g * amt) + (c2.g * (1-amt))),
    ((c1.b * amt) + (c2.b * (1-amt))),
    ((c1.a * amt) + (c2.a * (1-amt)))
  };
}

// x == l
// y == i
// i == j
Color low_color = {0, 0, 0, 0};
Color high_color = {255, 0, 180, 255};

void nn_draw(NN nn, int ox, int oy, int w, int h) {
  int cols = nn.count+1;
  int maxrows = 0;
  for (int i=0;i<cols;i++) {
    if (nn.as[i].cols > maxrows) {
       maxrows = nn.as[i].cols;
    }
  }
  float neuron_radius = fminf((float)h/ (maxrows*4), 25.0f);
  int pad_x = (w-neuron_radius*8) / (cols-1);
  int nn_x_start = ox + neuron_radius*4;
  int pad_y = h / (maxrows);
  DrawRectangleLines(ox, oy, w, h, RAYWHITE);
  DrawCircle(ox + neuron_radius*2, oy + neuron_radius*2, neuron_radius, low_color);
  DrawCircle(ox + neuron_radius*2, oy + neuron_radius*4, neuron_radius, high_color);
  for (size_t x = 0;x < cols; x++) {
    int rows = nn.as[x].cols;
    int nn_y_start = oy + (pad_y * ((maxrows) - rows+1)/2);
    Color color = {};
    for (size_t y = 0;y < rows;y++) {
      if (x == 0) {
        // first column is gray
        color = GRAY;
      } else {
        // second column and up are colored by their bias
        float amt = sigmoidf(MAT_AT(nn.bs[x-1], 0, y));
        color = mixColors(low_color, high_color, amt);
        //high_color.a = amt;
        //color = ColorAlphaBlend(low_color, high_color, WHITE);
      }
      DrawCircle(nn_x_start + (x * pad_x), nn_y_start + (y * pad_y), neuron_radius, color);
      if (x > 0) {
        int prev_rows = nn.as[x-1].cols;
        int prev_layer_y_start = oy + (pad_y * ((maxrows) - prev_rows+1)/2);
        for (size_t i = 0;i < prev_rows;i++) {
          float amt = sigmoidf(MAT_AT(nn.ws[x-1], i, y));
          Color linecolor = mixColors(low_color, high_color, amt);
          //high_color.a = amt;
          //Color linecolor = ColorAlphaBlend(low_color, high_color, WHITE);
          float thick = 1.0f;
          DrawLineEx((Vector2){nn_x_start + ((x-1) * pad_x), prev_layer_y_start + (i * pad_y)}, (Vector2){nn_x_start + x * pad_x, nn_y_start + (y * pad_y)}, thick, linecolor);
        }
      }
    }
  }
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max) {
  *min = FLT_MAX;
  *max = FLT_MIN;

  for (size_t i = 0;i<plot.count;i++) {
    if (*max < plot.items[i]) *max = plot.items[i];
    if (*min > plot.items[i]) *min = plot.items[i];
  }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh) {
  float min, max;
  cost_plot_minmax(plot, &min, &max);
  if (min > 0) min = 0;
  size_t n = plot.count;
  if (n < 100) n = 100;
  DrawRectangleLines(rx, ry, rw, rh, RAYWHITE);
  int steps = 20;
  for (int i=0;i<steps;i++) {
    int x = rx + (i * (rw / steps));
    DrawLine(x, ry, x, ry+rh, GRAY);
    int y = ry + (i * (rh / steps));
    DrawLine(rx, y, rx+rw, y, GRAY);
  }
  for (size_t i = 0;i+1<plot.count;i++) {
    float x = rx + (float)rw/n * i;
    float y = ry + (1-(plot.items[i] - min) / (max - min))*rh;
    float x2 = rx + (float)rw/n * (i+1);
    float y2 = ry + (1-(plot.items[i+1] - min) / (max - min))*rh;
    DrawLineEx((Vector2){x, y}, (Vector2){x2, y2}, rh*0.004, RED);
    DrawCircle(x, y, rh*0.004, RED);
  }

}

void ProcessInput() {
  if (IsKeyPressed(KEY_X)) {
    high_color = (Color){
      GetRandomValue(1, 255),
      GetRandomValue(1, 255),
      GetRandomValue(1, 255),
      GetRandomValue(1, 255)
    };
  }
  if (IsKeyPressed(KEY_C)) {
    low_color = (Color){
      GetRandomValue(1, 255),
      GetRandomValue(1, 255),
      GetRandomValue(1, 255),
      GetRandomValue(1, 255)
    };
  }
}

#define DA_INIT_CAP 256
#define da_append(da, item) do { \
      if ((da)->count >= (da)->capacity) { \
          (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2; \
          (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
          NN_ASSERT((da)->items != NULL && "NEED MORE RAM"); \
      } \
 \
      (da)->items[(da)->count++] = (item); \
    } while (0) \

#define MAX_ITER 5*1000

int main() {
  // seed input and output matrices based on adder truth table
  srand(time(0));
  size_t n = (1<<BITS);
  size_t rows = n*n;
  Mat ti = mat_alloc(rows, 2*BITS);
  Mat to = mat_alloc(rows, BITS + 1);
  for (size_t i = 0;i< ti.rows;i++) {
    size_t x = i/n;
    size_t y = i%n;
    size_t z = x + y;
    for (size_t j = 0;j < BITS;j++) {
      MAT_AT(ti, i, j)      = (x>>j)&1;
      MAT_AT(ti, i, j+BITS) = (y>>j)&1;
      MAT_AT(to, i, j)      = (z>>j)&1;
    }
    MAT_AT(to, i, BITS) = z >= n;
  }


  //size_t arch[] = {2*BITS, 4*BITS, 6*BITS, BITS+1};
  //size_t arch[] = {28*28, 16, 16, 10};
  //NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  //NN g = nn_alloc(arch,ARRAY_LEN(arch));
  NN nn;
  NN g;
  Arch arch = {0};
  float rate = 1.001;
  Cost_Plot plot = {0};
  float cost = 3.0f;

  float mincost = 10.0f;
  int iter = 0;
  InitWindow(width, height, "NN Raylib");
  SetWindowState(FLAG_WINDOW_RESIZABLE);
  while (!WindowShouldClose()) {
    // reset bounds based on current window size
    Vector2 dpi = GetWindowScaleDPI();
    width = GetRenderWidth()/dpi.x;
    height = GetRenderHeight()/dpi.y;
    if (iter == 0) {
      // clear our plot dynamic array
      plot.count = 0;
      plot.capacity = 0;
      plot.items = 0;
      // clear out arch dynamic array
      arch.count = 0;
      arch.capacity = 0;
      arch.items = 0;

      // Create new NN architecture of random size
      da_append(&arch, 2*BITS);
      size_t columns = GetRandomValue(1, 6);
      for (size_t i = 0;i<columns;i++) {
        da_append(&arch, GetRandomValue(2*BITS, 6*BITS));
      }
      da_append(&arch, BITS+1);

      nn = nn_alloc(arch.items, arch.count);
      g = nn_alloc(arch.items, arch.count);
      nn_rand(nn, -1, 1);

      cost = 3.0f; // reset arbitrary starting cost, just needs to be above 0.001
    }

    // run 10 cycles before drawing to screen to increase FPS
    for (size_t i=0;((i<10) && (i < MAX_ITER) && (cost >= 0.001f));i++) {
      nn_backprop(nn, g, ti, to);
      nn_learn(nn, g, rate);
      iter++;
      cost = nn_cost(nn, ti, to);

      if (iter%20 == 0) {
        da_append(&plot, cost);
      }
    }
    ProcessInput();
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("NN Raylib", 10, 10, 30, RAYWHITE);
    DrawText(TextFormat("Iter: %d", iter), 10, 50, 30, RAYWHITE);
    DrawText(TextFormat("Cost: %f", cost), 10, 90, 30, RAYWHITE);

    int rw, rh, rx, ry;
    rw = width/2;
    rh = height*((float)3/4);
    rx = width-rw;
    ry = height/2 - (float)rh/2;
    nn_draw(nn, rx, ry, rw, rh);
    rw = -10 + width/2;
    rh = height*((float)3/4);
    rx = 10;
    ry = height/2 - (float)rh/2;
    plot_cost(plot, rx, ry, rw, rh);

    EndDrawing();

    if (iter >= MAX_ITER || cost < 0.001f) {
      printf("Iters: %d\tFinal cost: %f\tArch: {", iter, cost);
      for (size_t i=0;i<nn.count+1;i++) {
        printf(" %zu", nn.as[i].cols);
      }
      printf(" }\n");
      iter = 0;
      // display validation data
      size_t fails = 0;
      for (size_t x = 0;x < n;x++) {
        for (size_t y = 0;y < n;y++) {
          size_t z = x + y;
          for (size_t j = 0;j< BITS;j++) {
            MAT_AT(NN_INPUT(nn), 0, j)      = (x>>j)&1;
            MAT_AT(NN_INPUT(nn), 0, j+BITS) = (y>>j)&1;
          }
          nn_forward(nn);
          if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
            if (z < n) {
              //printf("%zu + %zu = (OVERFLOW<>%zu)\n", x, y, z);
              fails++;
            }
          } else {
            size_t a = 0;
            for (size_t j = 0;j < BITS;j++) {
              size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
              a |= bit<<j;
            }
            if (z != a) {
              //printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
              fails++;
            }
          }
        }
      }
      //  end validation display
      //NN_PRINT(nn);
      //NN_PRINT(g);
      printf("%zu validation failures\n", fails);
      // free previous nn
      nn_free(nn);
      nn_free(g);
      free(plot.items);
      free(arch.items);
    }
  }
  CloseWindow();
  return 0;


  /*
  printf("cost = %f\n", nn_cost(nn, ti, to));
  for (size_t i = 0 ;i < 10*1000;i++) {
    nn_backprop(nn, g, ti, to);
    //nn_finite_diff(nn, g, rate, ti, to);
    nn_learn(nn, g, rate);
    printf("cost = %f\n", nn_cost(nn, ti, to));
  }

  return 0;
  */
}

