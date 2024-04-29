#define NN_IMPLEMENTATION
#include <stdio.h>
#include "nn.h"
#include <time.h>

#include "/usr/local/include/raylib.h"

#define BITS 4

float width = 1600;
float height = 1200;



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

void nn_draw(NN nn) {
  int cols = nn.count+1;
  //int startcol = (width - 40) / cols;
  int maxrows = 0;
  for (int i=0;i<cols;i++) {
    if (nn.as[i].cols > maxrows) {
       maxrows = nn.as[i].cols;
    }
  }
  float neuron_radius = fminf(height / (maxrows*4), 25.0f);
  int pad_x = width / (cols + 2);
  int nn_x_start = pad_x + (pad_x/2);
  int pad_y = height / ((maxrows + 2));
  DrawRectangleLines(pad_x, pad_y, width-pad_x*2, height-pad_y*2, RAYWHITE);
  DrawCircle(pad_x + neuron_radius*2, pad_y + neuron_radius*2, neuron_radius, low_color);
  DrawCircle(pad_x + neuron_radius*2, pad_y + neuron_radius*4, neuron_radius, high_color);
  for (size_t x = 0;x < cols; x++) {
    int rows = nn.as[x].cols;
    int nn_y_start = pad_y + ((pad_y/2) * ((maxrows+1) - rows));
    Color color = {};
    for (size_t y = 0;y < rows;y++) {
      if (x == 0) {
        // first column is gray
        color = GRAY;
      } else {
        // second column and up are colored by their bias
        float amt = sigmoidf(MAT_AT(nn.bs[x-1], 0, y));
        color = mixColors(low_color, high_color, amt);
      }
      DrawCircle(nn_x_start + x * pad_x, nn_y_start + (y * pad_y), neuron_radius, color);
      if (x > 0) {
        int prev_rows = nn.as[x-1].cols;
        int prev_layer_y_start = pad_y + ((pad_y/2) * ((maxrows+1) - prev_rows));
        for (size_t i = 0;i < prev_rows;i++) {
          float amt = sigmoidf(MAT_AT(nn.ws[x-1], i, y));
          Color linecolor = mixColors(low_color, high_color, amt);
          DrawLine(nn_x_start + ((x-1) * pad_x), prev_layer_y_start + (i * pad_y), nn_x_start + x * pad_x, nn_y_start + (y * pad_y), linecolor);
        }
      }
    }
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

int main() {
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
  size_t arch[] = {};
  //NN_PRINT(nn);
  float rate = 1;

  float mincost = 10.0f;
  int iter = 0;
  InitWindow(width, height, "NN Raylib");
  while (!WindowShouldClose()) {
    if (iter == 0) {
      //NN_PRINT(nn);
      // Create new NN architecture of random size
      size_t columns = GetRandomValue(1, 6);
      size_t arch[columns+2];
      arch[0] = 2*BITS;
      arch[columns+1] = BITS+1;
      for (size_t i = 1;i<columns+1;i++) {
        arch[i] = GetRandomValue(BITS, 10*BITS);
      }
      nn = nn_alloc(arch, ARRAY_LEN(arch));
      g = nn_alloc(arch, ARRAY_LEN(arch));
      nn_rand(nn, -1, 1);
    }
    ProcessInput();
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("NN Raylib", 10, 10, 30, RAYWHITE);
    DrawText(TextFormat("Iter: %d", iter), 10, 50, 30, RAYWHITE);
    DrawText(TextFormat("Cost: %f", nn_cost(nn, ti, to)), 10, 90, 30, RAYWHITE);
    nn_backprop(nn, g, ti, to);
    nn_learn(nn, g, rate);
    nn_draw(nn);
    EndDrawing();

    iter++;
    if (iter > 5*1000 || nn_cost(nn, ti, to) < 0.001f) {
      printf("Iters: %d\tFinal cost: %f\tArch: {", iter, nn_cost(nn, ti, to));
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
    }

  }
  CloseWindow();
  return 0;


  printf("cost = %f\n", nn_cost(nn, ti, to));
  for (size_t i = 0 ;i < 10*1000;i++) {
    nn_backprop(nn, g, ti, to);
    //nn_finite_diff(nn, g, rate, ti, to);
    nn_learn(nn, g, rate);
    printf("cost = %f\n", nn_cost(nn, ti, to));
  }

  return 0;
}

