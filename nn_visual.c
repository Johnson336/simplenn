#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"
#include "/usr/local/include/raylib.h"

float width = 1600;
float height = 1200;
const size_t BITS = 4;
size_t arch[] = {4, 4, 2, 1};

Color mixColors(Color c1, Color c2, float amt) {
  return (Color) {
    ((c1.r * amt) + (c2.r * (1-amt))),
    ((c1.g * amt) + (c2.g * (1-amt))),
    ((c1.b * amt) + (c2.b * (1-amt))),
    255
  };
}

void nn_draw(NN nn) {
  int cols = ARRAY_LEN(arch);
  //int startcol = (width - 40) / cols;
  int maxrows = 0;
  for (int i=0;i<cols;i++) {
    if (arch[i] > maxrows) {
       maxrows = arch[i];
    }
  }
  float neuron_radius = fminf(height / (maxrows*4), 25.0f);
  int pad_x = width / (cols + 2);
  int nn_x_start = pad_x + (pad_x/2);
  int pad_y = height / ((maxrows + 2));
  DrawRectangleLines(pad_x, pad_y, width-pad_x*2, height-pad_y*2, RAYWHITE);
  for (size_t x = 0;x < cols; x++) {
    int rows = arch[x];
    int nn_y_start = pad_y + ((pad_y/2) * ((maxrows+1) - rows));
    Color low_color = {0,0,255,255};
    Color high_color = {0, 255, 0, 255};
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
        int prev_rows = arch[x-1];
        int prev_layer_y_start = pad_y + ((pad_y/2) * ((maxrows+1) - prev_rows));
        for (size_t i = 0;i < prev_rows;i++) {
          float amt = sigmoidf(MAT_AT(nn.ws[x-1], y, i));
          Color linecolor = mixColors(low_color, high_color, amt);
          DrawLine(nn_x_start + ((x-1) * pad_x), prev_layer_y_start + (i * pad_y), nn_x_start + x * pad_x, nn_y_start + (y * pad_y), linecolor);
        }
      }
    }
  }


}

int main() {
  srand(time(0));
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, -1, 1);
  NN_PRINT(nn);
  InitWindow(width, height, "NN Raylib");
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawRectangleLines(0, 0, width, height, RAYWHITE);
    DrawText("NN Raylib", 10, 10, 30, RAYWHITE);
    nn_draw(nn);
    EndDrawing();
  }
  CloseWindow();
  
  return 0;
}


