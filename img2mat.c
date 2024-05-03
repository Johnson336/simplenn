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
size_t MAX_ITER = 10 * 1000;
size_t iter = 0;
float cost = 3.0f;
float rate = 1.0f;
bool paused = true;



void ProcessInput() {
  if (IsKeyPressed(KEY_SPACE)) {
    paused = !paused;
  }
}

int main(int argc, char **argv) {
  const char *program = args_shift(&argc, &argv);
  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
    fprintf(stderr, "ERROR: No input file given\n");
    return 1;
  }
  const char *img1_file_path = args_shift(&argc, &argv);
  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
    fprintf(stderr, "ERROR: No input file given\n");
    return 1;
  }
  const char *img2_file_path = args_shift(&argc, &argv);
  
  int img1_width, img1_height, img1_comp;
  uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
  if (img1_pixels == NULL) {
    fprintf(stderr, "ERROR: could not read image %s\n", img1_file_path);
    return 1;
  }
  if (img1_comp != 1) {
    fprintf(stderr, "ERROR: the image %s is %d bits image. Only 8 bit grayscale images are supported\n", img1_file_path, img1_comp*8);
    return 1;
  }

  printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp*8);

  int img2_width, img2_height, img2_comp;
  uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
  if (img2_pixels == NULL) {
    fprintf(stderr, "ERROR: could not read image %s\n", img2_file_path);
    return 1;
  }
  if (img2_comp != 1) {
    fprintf(stderr, "ERROR: the image %s is %d bits image. Only 8 bit grayscale images are supported\n", img2_file_path, img2_comp*8);
    return 1;
  }

  printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);

  size_t arch[] = {3, 12, 12, 1};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g = nn_alloc(arch, ARRAY_LEN(arch));

  // allocate training data
  // rows = total pixels in image, width * height
  // cols = architecture of neural network
  // 2 inputs = x, y  (coords of pixel)
  // 1 output = b     (brightness of grayscale pixel)
  Mat t = mat_alloc(img1_width*img1_height + img2_width*img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);


  // normalized coordinates from 0 - 1
  // x / width = 0-1
  // x = x/w
  // y / height = 0-1
  // y = y/h
  for (int y = 0; y < img1_height;y++) {
    for (int x = 0;x < img1_width;x++) {
      size_t i = y*img1_width + x;
      MAT_AT(t, i, 0) = (float)x / (img1_width - 1);
      MAT_AT(t, i, 1) = (float)y / (img1_height - 1);
      MAT_AT(t, i, 2) = 0.0f;
      MAT_AT(t, i, 3) = img1_pixels[i]/255.f;
    }
  }
  for (int y = 0; y < img2_height;y++) {
    for (int x = 0;x < img2_width;x++) {
      size_t i = img1_width*img1_height + y*img2_width + x;
      MAT_AT(t, i, 0) = (float)x / (img2_width - 1);
      MAT_AT(t, i, 1) = (float)y / (img2_height - 1);
      MAT_AT(t, i, 2) = 1.0f;
      MAT_AT(t, i, 3) = img2_pixels[y*img2_width + x]/255.f;
    }
  }

  Mat ti = {
    .rows = t.rows,
    .cols = NN_INPUT(nn).cols,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, 0),
 };

  Mat to = {
    .rows = t.rows,
    .cols = NN_OUTPUT(nn).cols,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, ti.cols),
  };

  //MAT_PRINT(ti);
  //MAT_PRINT(to);
  const char *mat_file_path = "img.mat";
  FILE *mat_output = fopen(mat_file_path, "wb");
  mat_save(mat_output, t);
  fclose(mat_output);

  InitWindow(width, height, "NN Img2Png");
  //SetWindowState(FLAG_WINDOW_RESIZABLE);

  size_t out_width = img1_width;
  size_t out_height = img1_height;
  uint8_t *out1_pixels = malloc(sizeof(*out1_pixels)*out_width*out_height);
  assert(out1_pixels != NULL);

  Image preview_image1 = GenImageColor(out_width, out_height, BLACK);
  Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

  Image input_image1 = LoadImage(img1_file_path);
  Texture2D input_texture1 = LoadTextureFromImage(input_image1);
  
  uint8_t *out2_pixels = malloc(sizeof(*out2_pixels)*out_width*out_height);
  assert(out2_pixels != NULL);

  Image preview_image2 = GenImageColor(out_width, out_height, BLACK);
  Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

  Image input_image2 = LoadImage(img2_file_path);
  Texture2D input_texture2 = LoadTextureFromImage(input_image2);

  nn_rand(nn, -1, 1);

  size_t batch_size = 28;
  size_t batch_count = (t.rows + batch_size - 1) / batch_size;
  size_t batches_per_frame = 100;
  size_t batch_begin = 0;
  float average_cost = 0.0f;

  while (!WindowShouldClose()) {

    ProcessInput();
    /*
    Vector2 dpi = GetWindowScaleDPI();
    width = GetRenderWidth()/dpi.x;
    height = GetRenderHeight()/dpi.y;
    */

    for (size_t i=0;i<batches_per_frame && iter < MAX_ITER && !paused;i++) {
      size_t size = batch_size;
      if (batch_begin + batch_size >= t.rows) {
        size = t.rows - batch_begin;
      }

      Mat batch_ti = {
        .rows = size,
        .cols = 3,
        .stride = t.stride,
        .es = &MAT_AT(t, batch_begin, 0),
      };

      Mat batch_to = {
        .rows = size,
        .cols = 1,
        .stride = t.stride,
        .es = &MAT_AT(t, batch_begin, batch_ti.cols),
      };
      
      
      nn_backprop(nn, g, batch_ti, batch_to);
      nn_learn(nn, g, rate);
      average_cost += nn_cost(nn, batch_ti, batch_to);
      batch_begin += batch_size;

      if (batch_begin >= t.rows) {
        iter++;
        average_cost = 0.0f;
        batch_begin = 0;
        mat_shuffle_rows(t);
      }
      if (iter%100==0) {
        for (size_t y = 0;y < out_height;y++) {
          for (size_t x = 0;x < out_width;x++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            ImageDrawPixel(&preview_image1, x, y, (Color){pixel, pixel, pixel, 255});
          }
        }
        //printf("%zu: cost = %f\n", iter, cost);
        Color *pixels1 = LoadImageColors(preview_image1);
        UpdateTexture(preview_texture1, pixels1);
        for (size_t y = 0;y < out_height;y++) {
          for (size_t x = 0;x < out_width;x++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
            MAT_AT(NN_INPUT(nn), 0, 2) = 1.0f;
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            ImageDrawPixel(&preview_image2, x, y, (Color){pixel, pixel, pixel, 255});
          }
        }
        //printf("%zu: cost = %f\n", iter, cost);
        Color *pixels2 = LoadImageColors(preview_image2);
        UpdateTexture(preview_texture2, pixels2);
      }
    }

    BeginDrawing();
    ClearBackground(BLACK);
    DrawFPS(width-100, 10);

    DrawText(TextFormat("Iter: %d", iter), 10, 50, 30, RAYWHITE);
    //DrawText(TextFormat("Cost: %f", 0), 10, 90, 30, RAYWHITE);

    DrawText("SimpleNN Generated", 530, 140, 30, WHITE);
    DrawText("Original", 150, 140, 30, WHITE);
    DrawTextureEx(input_texture1, (Vector2){20, 170}, 0, 10, WHITE);
    DrawTextureEx(preview_texture1, (Vector2){450, 170}, 0, 10, WHITE);
    DrawTextureEx(input_texture2, (Vector2){20, 400}, 0, 10, WHITE);
    DrawTextureEx(preview_texture2, (Vector2){450, 400}, 0, 10, WHITE);

    

    EndDrawing();

    if (iter == MAX_ITER && !paused) {
      iter = -1;

      /*
      for (size_t y = 0;y < img1_height;y++) {
        for (size_t x = 0;x < img1_width;x++) {
          uint8_t pixel = img1_pixels[y*img1_width + x];
          if (pixel > 0) {
            printf("%3u ", pixel);
          } else {
            printf("    ");
          }
        }
        printf("\n");
      }

      for (size_t y = 0;y < img1_height;y++) {
        for (size_t x = 0;x < img1_width;x++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (img1_width - 1);
          MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (img1_height - 1);
          MAT_AT(NN_INPUT(nn), 0, 2) = 1.0f;
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

      */

      for (size_t y = 0;y < out_height;y++) {
        for (size_t x = 0;x < out_width;x++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
          MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
          MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
          nn_forward(nn);
          uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
          out1_pixels[y*out_width + x] = pixel;
        }
      }


      const char *out_file_path = "img1out.png";
      if (!stbi_write_png(out_file_path, out_width, out_height, 1, out1_pixels, out_width*sizeof(*out1_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
      }

      printf("Generated %s from %s\n", out_file_path, img1_file_path);

      for (size_t y = 0;y < out_height;y++) {
        for (size_t x = 0;x < out_width;x++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = (float) x / (out_width - 1);
          MAT_AT(NN_INPUT(nn), 0, 1) = (float) y / (out_height - 1);
          MAT_AT(NN_INPUT(nn), 0, 2) = 1.0f;
          nn_forward(nn);
          uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
          out2_pixels[y*out_width + x] = pixel;
        }
      }


      out_file_path = "img2out.png";
      if (!stbi_write_png(out_file_path, out_width, out_height, 1, out2_pixels, out_width*sizeof(*out2_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
      }

      printf("Generated %s from %s\n", out_file_path, img2_file_path);
    }
  }

  CloseWindow();

  return 0;
}
