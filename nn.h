#ifndef __NN_H
#define __NN_H

// float d[] = {0, 1, 0, 1};
// Mat di = { .rows = 4, .cols = 2, .stride = 3, .es = d };
// Mat do = { .rows = 4, .cols = 1, .stride = 3, .es = d };
//
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#ifndef NN_FREE
#define NN_FREE free
#endif

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

// define the model
typedef struct {
  size_t count;
  Mat *ws; // array of weight matrices
  Mat *bs; // array of bias matrices
  Mat *as; // array of activation matrices is count + 1
} NN;



// Utility declares
#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
float rand_float();
float sigmoidf(float x);
float relu(float x);
float leaky_relu(float x);
float swish(float x);

// Mat function declares
Mat mat_alloc(size_t rows, size_t cols);
void mat_free(Mat m);
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in);
void mat_fill(Mat m, float x);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m, const char *name, size_t padding);
void mat_shuffle_rows(Mat m);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_sig(Mat m);
void mat_relu(Mat m);
void mat_leaky_relu(Mat m);
void mat_swish(Mat m);
#define MAT_PRINT(m) mat_print(m, #m, 0)
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

// NN function declares

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_free(NN nn);
void nn_zero(NN nn);
void nn_fill(NN nn, float n);
void nn_print(NN nn, const char *name, size_t padding);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN m, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
#define NN_PRINT(nn) nn_print(nn, #nn, 0)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]


#endif // __NN_H

#ifdef NN_IMPLEMENTATION

float rand_float() {
  return (float)rand() / (float)RAND_MAX;
}

float swish(float x) {
  return x / (1.f + expf(-x));
}

float leaky_relu(float x) {
  return fmaxf(0.1f*x, x);
}

float relu(float x) {
  return fmaxf(0, x);
  // return (x > 0 ? x : 0);
}
float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols) {
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = m.cols;
  m.es = (float *)NN_MALLOC(sizeof(*m.es)*rows*cols);
  NN_ASSERT(m.es != NULL);
  return m;
}

void mat_free(Mat m) {
  NN_FREE(m.es);
}

void mat_save(FILE *out, Mat m) {
  const char *magic = "nn.h.mat";
  fwrite(magic, strlen(magic), 1, out);
  fwrite(&m.rows, sizeof(m.rows), 1, out);
  fwrite(&m.cols, sizeof(m.cols), 1, out);
  for (size_t i = 0;i < m.rows;i++) {
    size_t n = fwrite(&MAT_AT(m, i, 0), sizeof(*m.es), m.cols, out);
    while (n < m.cols && !ferror(out)) {
      size_t j = fwrite(m.es + n, sizeof(*m.es), m.cols - n, out);
      n += i;
    }
  }
}
Mat mat_load(FILE *in) {

  uint64_t magic;
  fread(&magic, sizeof(magic), 1, in);
  NN_ASSERT(magic == 0x74616d2e682e6e6e);
  
  size_t rows, cols;
  fread(&rows, sizeof(rows), 1, in);
  fread(&cols, sizeof(cols), 1, in);
  Mat m = mat_alloc(rows, cols);
  size_t n = fread(m.es, sizeof(*m.es), rows*cols, in);
  while (n < rows*cols && !ferror(in)) {
  size_t k = fread(m.es, sizeof(*m.es) + n, rows*cols - n, in);
    n += k;
  }
  return m;

}

void mat_dot(Mat dst, Mat a, Mat b) {
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == b.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; k++) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }  
}

void mat_fill(Mat m, float x) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = x;
    }
  }
}

Mat mat_row(Mat m, size_t row) {
  return (Mat){
    .rows = 1,
    .cols = m.cols,
    .stride = m.stride,
    .es = &MAT_AT(m, row, 0),
  };
}

void mat_copy(Mat dst, Mat src) {
  NN_ASSERT(dst.rows == src.rows);
  NN_ASSERT(dst.cols == src.cols);
  for (size_t i = 0;i < dst.rows;i++) {
    for (size_t j = 0;j < dst.cols;j++) {
       MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_sum(Mat dst, Mat a) {
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == a.cols);
  for (size_t i = 0;i < dst.rows; i++) {
    for (size_t j = 0;j < dst.cols; j++) {
       MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_print(Mat m, const char *name, size_t padding) {
  printf("%*s%s = [\n", (int) padding, "", name);
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
       printf("%*s    %f ", (int)padding, "", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int) padding, "");
}

void mat_rand(Mat m, float low, float high) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}

void mat_relu(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = relu(MAT_AT(m, i, j));
    }
  }
}

void mat_leaky_relu(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = relu(MAT_AT(m, i, j));
    }
  }
}

void mat_swish(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = swish(MAT_AT(m, i, j));
    }
  }
}

void mat_sig(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}


NN nn_alloc(size_t *arch, size_t arch_count) {
  // define the model
  NN_ASSERT(arch_count > 0);
  NN nn;
  nn.count = arch_count - 1;

  nn.ws = (Mat *)NN_MALLOC(sizeof(*nn.ws)*nn.count);
  NN_ASSERT(nn.ws != NULL);
  nn.bs = (Mat *)NN_MALLOC(sizeof(*nn.bs)*nn.count);
  NN_ASSERT(nn.bs != NULL);
  nn.as = (Mat *)NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
  NN_ASSERT(nn.as != NULL);

  nn.as[0] = mat_alloc(1, arch[0]);
  for (size_t i = 1;i < arch_count;i++) {
    nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = mat_alloc(1, arch[i]);
    nn.as[i]   = mat_alloc(1, arch[i]);
  }

  return nn;
}

void nn_free(NN nn) {
  size_t count = nn.count;

  mat_free(nn.as[0]);
  for (size_t i = 1;i < count+1;i++) {
    mat_free(nn.ws[i-1]);
    mat_free(nn.bs[i-1]);
    mat_free(nn.as[i]);
  }
  NN_FREE(nn.ws);
  //NN_ASSERT(nn.ws == NULL);
  NN_FREE(nn.bs);
  //NN_ASSERT(nn.bs == NULL);
  NN_FREE(nn.as);
  //NN_ASSERT(nn.as == NULL);
}

void nn_print(NN nn, const char *name, size_t padding) {
  char buf[256];
  printf("%*s%s = [\n", (int) padding, "", name);
  Mat *ws = nn.ws;
  Mat *bs = nn.bs;
  for (size_t i = 0;i<nn.count;i++) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    mat_print(bs[i], buf, 4);
  }
  printf("%*s]\n", (int) padding, "");
}


void nn_zero(NN nn) {
  for (size_t i = 0;i<nn.count;i++) {
    mat_fill(nn.ws[i], 0);
    mat_fill(nn.bs[i], 0);
    mat_fill(nn.as[i], 0);
  }
  mat_fill(nn.as[nn.count], 0);
}
void nn_fill(NN nn, float n) {
  for (size_t i = 0;i<nn.count;i++) {
    mat_fill(nn.ws[i], n);
    mat_fill(nn.bs[i], n);
    mat_fill(nn.as[i], n);
  }
  mat_fill(nn.as[nn.count], n);
}

void nn_rand(NN nn, float low, float high) {
  for (size_t i = 0;i<nn.count;i++) {
    mat_rand(nn.ws[i], low, high);
    mat_rand(nn.bs[i], low, high);
  }
}

void nn_forward(NN nn) {
  for (size_t i = 0;i < nn.count;i++) {
    mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
    mat_sum(nn.as[i+1], nn.bs[i]);
    mat_sig(nn.as[i+1]);
    //mat_relu(nn.as[i+1]);
  }
}

float nn_cost(NN nn, Mat ti, Mat to) {
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
  size_t n = ti.rows;

  float c = 0;
  for (size_t i = 0;i< n;i++) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);
    mat_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    size_t q = to.cols;
    for (size_t j = 0;j < q;j++) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      c += d*d;
    }
  }


  return c/n;

}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to) {

  float saved;
  float c = nn_cost(nn, ti, to);
  for (size_t i = 0;i < nn.count;i++) {
    for (size_t j = 0;j < nn.ws[i].rows;j++) {
      for (size_t k = 0;k < nn.ws[i].cols;k++) {
        saved = MAT_AT(nn.ws[i], j, k);
        MAT_AT(nn.ws[i], j, k) += eps;
        MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
        MAT_AT(nn.ws[i], j, k) = saved;
      }
    }

    for (size_t j = 0;j < nn.bs[i].rows;j++) {
      for (size_t k = 0;k < nn.bs[i].cols;k++) {
        saved = MAT_AT(nn.bs[i], j, k);
        MAT_AT(nn.bs[i], j, k) += eps;
        MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
        MAT_AT(nn.bs[i], j, k) = saved;
      }
    }
  }
}


void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
  NN_ASSERT(ti.rows == to.rows);
  size_t n = ti.rows;
  NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

  nn_zero(g);

  // i - current sample
  // l - current layer
  // j - current activation
  // k - previous activation

  for (size_t i = 0;i<n;i++) {
    mat_copy(NN_INPUT(nn), mat_row(ti, i));
    nn_forward(nn);

    for (size_t j = 0;j <= nn.count;j++) {
      mat_fill(g.as[j], 0);
    }

    for (size_t j = 0;j < to.cols;j++) {
      MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
    }

    for (size_t l = nn.count;l > 0;l--) {
      for (size_t j = 0;j < nn.as[l].cols;j++) {
        float a = MAT_AT(nn.as[l], 0, j);
        float da = MAT_AT(g.as[l], 0, j);
        MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1-a);
        for (size_t k = 0;k < nn.as[l-1].cols;k++) {
          // j - weight matrix col
          // k - weight matrix row
          float pa = MAT_AT(nn.as[l-1], 0, k);
          float w = MAT_AT(nn.ws[l-1], k, j);
          MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1-a)*pa;
          MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1-a)*w;
        }
      }
    }
  }

  for (size_t i = 0;i < g.count;i++) {
    for (size_t j = 0;j< g.ws[i].rows;j++) {
      for (size_t k = 0;k < g.ws[i].cols;k++) {
        MAT_AT(g.ws[i], j, k) /= n;
      }
    }
    for (size_t j = 0;j< g.bs[i].rows;j++) {
      for (size_t k = 0;k < g.bs[i].cols;k++) {
        MAT_AT(g.bs[i], j, k) /= n;
      }
    }
  }
}


void nn_learn(NN nn, NN g, float rate) {
  for (size_t i = 0;i < nn.count;i++) {
    for (size_t j = 0;j < nn.ws[i].rows;j++) {
      for (size_t k = 0;k < nn.ws[i].cols;k++) {
        MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
      }
    }

    for (size_t j = 0;j < nn.bs[i].rows;j++) {
      for (size_t k = 0;k < nn.bs[i].cols;k++) {
        MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
      }
    }
  } 
}

// 5 - 2 = hat
// 0 1 2 3 4
// 5 6 4 3 7
//     ^ 

void mat_shuffle_rows(Mat m) {
  for (size_t i = 0;i < m.rows;i++) {
    size_t j = i + rand()%(m.rows - i);
    if (i != j) {
      for (size_t k = 0;k < m.cols;k++) {
        float t = MAT_AT(m, i, k);
        MAT_AT(m, i, k) = MAT_AT(m, j, k);
        MAT_AT(m, j, k) = t;
      }
    }
  }
}

#ifdef NN_ENABLE_GYM
#include "/usr/local/include/raylib.h"

typedef struct {
  float *items;
  size_t count;
  size_t capacity;
} Cost_Plot;

Color mixColors(Color c1, Color c2, float amt);
void nn_draw(NN nn, int rx, int ry, int rw, int rh);
void cost_plot_minmax(Cost_Plot plot, float *min, float *max);
void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh);

Color mixColors(Color c1, Color c2, float amt) {
  return (Color) {
    ((c1.r * amt) + (c2.r * (1-amt))),
    ((c1.g * amt) + (c2.g * (1-amt))),
    ((c1.b * amt) + (c2.b * (1-amt))),
    ((c1.a * amt) + (c2.a * (1-amt)))
  };
}

Color low_color = {0, 0, 0, 0};
Color high_color = {255, 0, 180, 255};

void nn_draw(NN nn, int rx, int ry, int rw, int rh) {
  int cols = nn.count+1;
  int maxrows = 0;
  for (int i=0;i<cols;i++) {
    if (nn.as[i].cols > maxrows) {
       maxrows = nn.as[i].cols;
    }
  }
  float neuron_radius = fminf((float)rh/ (maxrows*4), 25.0f);
  int pad_x = (rw-neuron_radius*8) / (cols-1);
  int nn_x_start = rx + neuron_radius*4;
  int pad_y = rh / (maxrows);
  DrawRectangleLines(rx, ry, rw, rh, RAYWHITE);
  DrawCircle(rx + neuron_radius*2, ry + neuron_radius*2, neuron_radius, low_color);
  DrawCircle(rx + neuron_radius*2, ry + neuron_radius*4, neuron_radius, high_color);
  for (size_t x = 0;x < cols; x++) {
    int rows = nn.as[x].cols;
    int nn_y_start = ry + ((float)pad_y * ((maxrows) - rows+1)/2);
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
        int prev_layer_y_start = ry + ((float)pad_y * ((maxrows) - prev_rows+1)/2);
        for (size_t i = 0;i < prev_rows;i++) {
          float amt = sigmoidf(MAT_AT(nn.ws[x-1], i, y));
          Color linecolor = mixColors(low_color, high_color, amt);
          //high_color.a = amt;
          //Color linecolor = ColorAlphaBlend(low_color, high_color, WHITE);
          float thick = 1.0f;
          DrawLineEx((Vector2){nn_x_start + (((float)x-1) * pad_x), prev_layer_y_start + ((float)i * pad_y)}, (Vector2){nn_x_start + (float)x * pad_x, nn_y_start + ((float)y * pad_y)}, thick, linecolor);
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

#endif // NN_ENABLE_GYM

#endif // NN_IMPLEMENTATION
