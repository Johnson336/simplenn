#include <string.h>
#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"
#include "time.h"
#define SV_IMPLEMENTATION
#include "sv.h"

float width = 800;
float height = 600;
// pause will stop program from repeating forever
bool paused = true;

typedef struct {
  size_t *items;
  size_t count;
  size_t capacity;
} Arch;

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
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    Vector2 mouseDelta = GetMouseDelta();
    width += mouseDelta.x;
    height += mouseDelta.y;
    SetWindowSize(width, height);
  }
  if (IsKeyPressed(KEY_SPACE)) {
    paused = !paused;
  }
}

char *args_shift(int *argc, char ***argv) {
  assert(*argc > 0);
  char *result = **argv;
  (*argc) -= 1;
  (*argv) += 1;
  return result;
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

int main(int argc, char **argv) {
  const char *program = args_shift(&argc, &argv);
  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
    fprintf(stderr, "ERROR: No arch file provided\n");
    return 1;
  }
  const char *arch_file_path = args_shift(&argc, &argv);

  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
    fprintf(stderr, "ERROR: No data file provided\n");
    return 1;    
  }
  const char *data_file_path = args_shift(&argc, &argv);

  int buffer_len = 0;
  unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);

  if (buffer == NULL) {
    return 1;
  }

  Arch arch = {0};

  // fill arch dynamic array from external arch file
  String_View content = sv_from_parts((const char *)buffer, buffer_len);
  content = sv_trim_left(content);
  while (content.count > 0 && isdigit(content.data[0])) {
    int x = sv_chop_u64(&content);
    da_append(&arch, x);
    content = sv_trim_left(content);
  }

  // fill input and output matrices from external data file
  FILE *in = fopen(data_file_path, "rb");
  if (in == NULL) {
    fprintf(stderr, "ERROR: could not read file %s\n", data_file_path);
    return 1;
  }
  Mat t = mat_load(in);
  fclose(in);


  NN_ASSERT(arch.count > 1);
  size_t in_size = arch.items[0];
  size_t out_size = arch.items[arch.count-1];
  NN_ASSERT(t.cols == (in_size + out_size));


  Mat ti = {
    .rows = t.rows,
    .cols = in_size,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, 0),
  };

  Mat to = {
    .rows = t.rows,
    .cols = out_size,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, in_size),
  };

  //MAT_PRINT(ti);
  //MAT_PRINT(to);

  InitWindow(width, height, "NN Gym");
  //SetWindowState(FLAG_WINDOW_RESIZABLE);

  NN nn = nn_alloc(arch.items, arch.count);
  NN g = nn_alloc(arch.items, arch.count);
  nn_rand(nn, -1, 1);
  //float cost = 3.0f;
  
  float rate = 1.0f;
  Cost_Plot plot = {0};

  size_t iter = 0;


  bool header_flash_visible = true;
  float header_flash_delay = 1.0f;
  float header_flash_timer = 0.0f;
  const char *header_text = "NN Training...";
  float header_width = MeasureText(header_text, 30);

  size_t batch_size = 28;
  size_t batch_count = (t.rows + batch_size - 1) / batch_size;
  size_t batches_per_frame = 100;
  size_t batch_begin = 0;
  float average_cost = 0.0f;


  while (!WindowShouldClose()) {
    // reset bounds based on current window size
    Vector2 dpi = GetWindowScaleDPI();
    width = GetRenderWidth()/dpi.x;
    height = GetRenderHeight()/dpi.y;
    // check for -1 here to avoid triggering on the first cycle
    if (iter == -1 && !paused) {
      iter = 0;
      // clear our plot dynamic array
      plot.count = 0;
      plot.capacity = 0;
      plot.items = 0;
      // clear out arch dynamic array
      arch.count = 0;
      arch.capacity = 0;
      arch.items = 0;

      // Create new NN architecture of random size
      da_append(&arch, in_size);
      size_t columns = GetRandomValue(1, 6);
      for (size_t i = 0;i<columns;i++) {
        da_append(&arch, GetRandomValue(3*in_size, 15*in_size));
      }
      da_append(&arch, out_size);

      nn = nn_alloc(arch.items, arch.count);
      g = nn_alloc(arch.items, arch.count);
      nn_rand(nn, -1, 1);

      average_cost = 0.0f; // reset arbitrary starting cost, just needs to be above 0.001
    }
    // run 10 cycles before drawing to screen to increase FPS
    for (size_t i=0;((i < batches_per_frame) && (iter < MAX_ITER) && !paused);i++) {

      size_t size = batch_size;
      if (batch_begin + batch_size >= t.rows) {
        size = t.rows - batch_begin;
      }

      Mat batch_ti = {
        .rows = size,
        .cols = NN_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, batch_begin, 0),
      };

      Mat batch_to = {
        .rows = size,
        .cols = NN_OUTPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, batch_begin, batch_ti.cols),
      };
      
      
      nn_backprop(nn, g, batch_ti, batch_to);
      nn_learn(nn, g, rate);
      average_cost += nn_cost(nn, batch_ti, batch_to);
      batch_begin += batch_size;

      if (batch_begin >= t.rows) {
        iter++;
        da_append(&plot, average_cost/batch_count);
        average_cost = 0.0f;
        batch_begin = 0;
        mat_shuffle_rows(t);
      }
    }
    ProcessInput();
    BeginDrawing();
    ClearBackground(BLACK);
    DrawFPS(width-100, 10);

    if (!paused) {
      header_flash_timer += GetFrameTime();
    }
    if (header_flash_timer >= header_flash_delay) {
      header_flash_timer = 0.0f;
      header_flash_visible = !header_flash_visible;
    }
    if (header_flash_visible) {
      DrawText(header_text, width/2 - header_width/2, 10, 30, RAYWHITE);
    }
    DrawText(TextFormat("Iter: %d", iter), 10, 50, 20, RAYWHITE);
    DrawText(TextFormat("Cost: %f", (plot.count > 0 ? plot.items[plot.count-1] : 0)), 10, 70, 20, RAYWHITE);
    int rw, rh, rx, ry;
    rw = width/2;
    rh = height*((float)2/3);
    rx = width-rw;
    ry = height/2 - (float)rh/2;
    nn_draw(nn, rx, ry, rw, rh);
    rw = width/2;
    rh = height*((float)2/3);
    rx = 0;
    ry = height/2 - (float)rh/2;
    plot_cost(plot, rx, ry, rw, rh);
    EndDrawing();

    if (iter >= MAX_ITER) {
      printf("Iters: %zu\tFinal cost: %f\tArch: {", iter, (plot.count > 0 ? plot.items[plot.count-1] : 0));
      for (size_t i=0;i<nn.count+1;i++) {
        printf(" %zu", nn.as[i].cols);
      }
      printf(" }\n");
      iter = -1;
      // free allocated memory
      nn_free(nn);
      nn_free(g);
      free(plot.items);
      free(arch.items);
    }
  }


  return 0;
}
