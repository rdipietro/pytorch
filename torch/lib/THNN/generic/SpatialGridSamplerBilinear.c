#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialGridSamplerBilinear.c"
#else

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )

static inline void THNN_(SpatialGridSamplerBilinear_shapeCheck)
     (THTensor *input, THTensor *grid, THTensor *gradOutput) {
  THNN_ARGCHECK(input->nDimension == 4, 2, input,
		"4D input tensor expected but got: %s");
  THNN_ARGCHECK(grid->nDimension == 4, 2, grid,
		"4D grid tensor expected but got: %s");

  int nbatch   = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int height   = THTensor_(size)(input, 2);
  int width    = THTensor_(size)(input, 3);

  THNN_CHECK_DIM_SIZE(grid, 4, 0, nbatch);
  THNN_CHECK_DIM_SIZE(grid, 4, 1, height);
  THNN_CHECK_DIM_SIZE(grid, 4, 2, width);
  THNN_CHECK_DIM_SIZE(grid, 4, 3, 2);
  
  if (gradOutput != NULL) {
    THNN_CHECK_SHAPE(input, gradOutput);
  }
}

#define SAFE_GET(input, x, y, n, c, H, W) x >= 0 && x < W && y >=0 \
    && y < H ? THTensor_fastGet4d(input, n, c, y, x) : 0

TH_API void THNN_(SpatialGridSamplerBilinear_updateOutput)(
	  THNNState *state,
	  THTensor *input,
	  THTensor *grid,
	  THTensor *output) {

  THNN_(SpatialGridSamplerBilinear_shapeCheck)(input, grid, NULL);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int H = THTensor_(size)(input, 2);
  int W = THTensor_(size)(input, 3);
	  
  // resize output to the same shape as input
  THTensor_(resize4d)(output, N, C, H, W);
  real *idata = THTensor_(data)(input);
  real *gdata = THTensor_(data)(input);
  real *odata = THTensor_(data)(output);

  // loop over each output pixel
  int n, h, w, c;
#pragma omp parallel for private(n, h, w, c)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
	// get the corresponding input x, y co-ordinates from grid
	real ix = THTensor_fastGet4d(grid, n, h, w, 0);
	real iy = THTensor_fastGet4d(grid, n, h, w, 1);

	// normalize ix, iy from [-1, 1] to [0, H-1] & [0, W-1]
	ix = ((ix + 1) / 2) * (W-1);
	iy = ((iy + 1) / 2) * (H-1);

	// get NE, NW, SE, SW pixel values from (x, y)
	int ix_nw = floor(ix);
	int iy_nw = floor(iy);
	int ix_ne = ix_nw + 1;
	int iy_ne = iy_nw;
	int ix_sw = ix_nw;
	int iy_sw = iy_nw + 1;
	int ix_se = ix_nw + 1;
	int iy_se = iy_nw + 1;

	// get surfaces to each neighbor:
	real nw = (ix_se - ix)    * (iy_se - iy);
	real ne = (ix    - ix_sw) * (iy_sw - iy);
	real sw = (ix_ne - ix)    * (iy    - iy_ne);
	real se = (ix    - ix_nw) * (iy    - iy_nw);
	  
	// calculate bilinear weighted pixel value and set output pixel
	for (c = 0; c < C; ++c) {
	  //   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
	  // + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
	  real nw_val = SAFE_GET(input, ix_nw, iy_nw, n, c, H, W);
	  real ne_val = SAFE_GET(input, ix_ne, iy_ne, n, c, H, W);
	  real sw_val = SAFE_GET(input, ix_sw, iy_sw, n, c, H, W);
	  real se_val = SAFE_GET(input, ix_se, iy_se, n, c, H, W);
	  real out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
	  THTensor_fastSet4d(output, n, c, h, w, out_val);
	}
      }
    }
  }
}

TH_API void THNN_(SpatialGridSamplerBilinear_updateGradInput)(
	  THNNState *state,
	  THTensor *input, THTensor *gradInput,
	  THTensor *grid, THTensor *gradGrid,
	  THTensor *gradOutput) {

  THNN_(SpatialGridSamplerBilinear_shapeCheck)(input, grid, gradOutput);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int H = THTensor_(size)(input, 2);
  int W = THTensor_(size)(input, 3);

  THTensor_(resize4d)(gradInput, N, C, H, W);
  THTensor_(resize4d)(gradGrid, N, H, W, 2);
  
  THTensor_(zero)(gradInput);
  THTensor_(zero)(gradGrid);
  

}

#undef MIN
#undef SAFE_GET

#endif
