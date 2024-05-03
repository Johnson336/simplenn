#include <assert.h>
#include <stdio.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#include "/usr/local/include/raylib.h"


char *args_shift(int *argc, char ***argv) {
  assert(*argc > 0);
  char *result = **argv;
  (*argc) -= 1;
  (*argv) += 1;
  return result;
}

float width = 1200;
float height = 800;
float MAX_ITER = 100 * 1000;
size_t iter = 0;
float cost = 3.0f;
bool paused = true;

void ProcessInput() {
  if (IsKeyPressed(KEY_SPACE)) {
    paused = !paused;
  }
}

int main(int argc, char **argv) {
  const char *program = args_shift(&argc, &argv);
  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <input.png>\n", program);
    fprintf(stderr, "ERROR: No input file given\n");
    return 1;
  }
  const char *img_file_path = args_shift(&argc, &argv);
  
  int img_width, img_height, img_comp;
  uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);
  if (img_pixels == NULL) {
    fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
    return 1;
  }
  if (img_comp != 1) {
    fprintf(stderr, "ERROR: the image %s is %d bits image. Only 8 bit grayscale images are supported\n", img_file_path, img_comp*8);
    return 1;
  }

  printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);


  // allocate training data
  // rows = total pixels in image, width * height
  // cols = architecture of neural network
  // 2 inputs = x, y  (coords of pixel)
  // 1 output = b     (brightness of grayscale pixel)
  Mat t = mat_alloc(img_width*img_height, 3);


  // normalized coordinates from 0 - 1
  // x / width = 0-1
  // x = x/w
  // y / height = 0-1
  // y = y/h
  for (int y = 0; y < img_height;y++) {
    for (int x = 0;x < img_width;x++) {
      size_t i = y*img_width + x;
      MAT_AT(t, i, 0) = (float)x/(img_width - 1);
      MAT_AT(t, i, 1) = (float)y / (img_height-1);
      MAT_AT(t, i, 2) = img_pixels[i]/255.f;
    }
  }

  Mat ti = {
    .rows = t.rows,
    .cols = 2,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, 0),
  };

  Mat to = {
    .rows = t.rows,
    .cols = 1,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, ti.cols),
  };

  MAT_PRINT(ti);
  MAT_PRINT(to);

  InitWindow(width, height, "NN Img2Png");
  //SetWindowState(FLAG_WINDOW_RESIZABLE);

  size_t out_width = 512;
  size_t out_height = 512;
  uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
  assert(out_pixels != NULL);

  Image preview_image = GenImageColor(out_width, out_height, BLACK);
  Texture2D preview_texture = LoadTextureFromImage(preview_image);

  Image input_image = LoadImage(img_file_path);
  Texture2D input_texture = LoadTextureFromImage(input_image);

  float rate = 1.0f;
  size_t arch[] = {2, 6, 6, 8, 8, 1};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, -1, 1);

  while (!WindowShouldClose()) {

    ProcessInput();
    /*
    Vector2 dpi = GetWindowScaleDPI();
    width = GetRenderWidth()/dpi.x;
    height = GetRenderHeight()/dpi.y;
    */

    for (size_t i=0;i<10 && iter < MAX_ITER && !paused;i++) {
      nn_backprop(nn, g, ti, to);
      nn_learn(nn, g, rate);
      iter++;
      cost = nn_cost(nn, ti, to);
      if (iter%100==0) {
        for (size_t y = 0;y < out_height;y++) {
          for (size_t x = 0;x < out_width;x++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            ImageDrawPixel(&preview_image, x, y, (Color){pixel, pixel, pixel, 255});
          }
        }
        //printf("%zu: cost = %f\n", iter, cost);
        Color *pixels = LoadImageColors(preview_image);
        UpdateTexture(preview_texture, pixels);
      }
    }

    BeginDrawing();
    ClearBackground(BLACK);
    DrawFPS(width-100, 10);

    DrawText(TextFormat("Iter: %d", iter), 10, 50, 30, RAYWHITE);
    DrawText(TextFormat("Cost: %f", cost), 10, 90, 30, RAYWHITE);

    DrawText("SimpleNN Generated", 630, 140, 30, WHITE);
    DrawText("Original", 250, 140, 30, WHITE);
    DrawTextureEx(input_texture, (Vector2){20, 170}, 0, 20, WHITE);
    DrawTextureEx(preview_texture, (Vector2){550, 200}, 0, 1, WHITE);

    

    EndDrawing();

    if (iter == MAX_ITER && !paused) {

      for (size_t y = 0;y < img_height;y++) {
        for (size_t x = 0;x < img_width;x++) {
          uint8_t pixel = img_pixels[y*img_width + x];
          if (pixel > 0) {
            printf("%3u ", pixel);
          } else {
            printf("    ");
          }
        }
        printf("\n");
      }

      for (size_t y = 0;y < img_height;y++) {
        for (size_t x = 0;x < img_width;x++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (img_width - 1);
          MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (img_height - 1);
          nn_forward(nn);
          uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
          if (pixel > 0) {
          printf("%3u ", pixel);
          } else {
            printf("    ");
          }
        }
        printf("\n");
      }


      for (size_t y = 0;y < out_height;y++) {
        for (size_t x = 0;x < out_width;x++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
          MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
          nn_forward(nn);
          uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
          out_pixels[y*out_width + x] = pixel;
        }
      }


      const char *out_file_path = "imgout.png";
      if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
      }

      printf("Generated %s from %s\n", out_file_path, img_file_path);
    }
  }

  CloseWindow();

  return 0;
}
