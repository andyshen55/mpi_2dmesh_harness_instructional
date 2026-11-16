// Part 2 correctness checks. Compile separately with the MPI C++ wrapper.
// Include the actual implementation; do not maintain a second copy of its code.
#define main hw6_harness_main
#include "../../mpi_2dmesh.cpp"
#undef main
#include <fstream>
#include <cstdlib>
#include <cmath>

namespace {
int failures = 0;

void check(bool condition, const char *message, int rank)
{
   if (!condition) {
      if (failures < 10) fprintf(stderr, "FAIL rank %d: %s\n", rank, message);
      ++failures;
   }
}

float recognizable(int row, int col) { return row*1000 + col; }

// Nonzero offsets and different strides in both directions; untouched cells
// must retain the sentinel. The larger message is roughly 456 KB of floats.
void transfers(int rank)
{
   const int cases[][6] = {{8, 5, 2, 1, 3, 2}, {513, 257, 7, 11, 489, 233}};
   for (const auto &c : cases) {
      int sw=c[0], sh=c[1], sx=c[2], sy=c[3], w=c[4], h=c[5];
      int dw=sw+5, dh=sh+4, rw=sw+2, rh=sh+3;
      const float sentinel=-9876.0f;
      if (rank == 0) {
         vector<float> source(sw*sh), returned(rw*rh, sentinel);
         for (int y=0; y<sh; ++y)
            for (int x=0; x<sw; ++x) source[y*sw+x]=recognizable(y,x);
         sendStridedBuffer(source.data(), sw, sh, sx, sy, w, h, 0, 1);
         recvStridedBuffer(returned.data(), rw, rh, 1, 3, w, h, 1, 0);
         for (int y=0; y<rh; ++y)
            for (int x=0; x<rw; ++x) {
               bool inside=x>=1 && x<1+w && y>=3 && y<3+h;
               float expected=inside ? recognizable(sy+y-3,sx+x-1) : sentinel;
               check(returned[y*rw+x]==expected, "round-trip rectangle/sentinel", rank);
            }
      } else {
         vector<float> destination(dw*dh, sentinel);
         recvStridedBuffer(destination.data(), dw, dh, 4, 2, w, h, 0, 1);
         for (int y=0; y<dh; ++y)
            for (int x=0; x<dw; ++x) {
               bool inside=x>=4 && x<4+w && y>=2 && y<2+h;
               float expected=inside ? recognizable(sy+y-2,sx+x-4) : sentinel;
               check(destination[y*dw+x]==expected, "receive rectangle/sentinel", rank);
            }
         sendStridedBuffer(destination.data(), dw, dh, 4, 2, w, h, 1, 0);
      }
   }
}

// Independent unrolled double reference for the small known-answer cases.
double reference_pixel(const vector<float> &s, int w, int h, int x, int y)
{
   if (x==0 || y==0 || x==w-1 || y==h-1) return 0;
   auto at=[&](int dy, int dx) { return double(s[(y+dy)*w+x+dx]); };
   double gx=at(-1,-1)+2*at(0,-1)+at(1,-1)-at(-1,1)-2*at(0,1)-at(1,1);
   double gy=at(-1,-1)+2*at(-1,0)+at(-1,1)-at(1,-1)-2*at(1,0)-at(1,1);
   return std::sqrt(gx*gx+gy*gy);
}

void distributed_case(AppState &as, const vector<float> &reference,
                      const char *label, bool small)
{
   const int w=as.global_mesh_size[0], h=as.global_mesh_size[1], rank=as.myrank;
   vector<vector<Tile2D>> tiles;
   computeMeshDecomposition(&as, &tiles);
   vector<int> owners(as.nranks, 0), coverage(small ? w*h : 0, 0);
   for (const auto &row : tiles)
      for (const auto &t : row) {
         check(t.tileRank>=0 && t.tileRank<as.nranks, "tile owner range", rank);
         if (t.tileRank>=0 && t.tileRank<as.nranks) ++owners[t.tileRank];
         check(t.width>0 && t.height>0, "positive core dimensions", rank);
         check(t.xloc-t.ghost_xmin==std::max(0,t.xloc-1), "left halo", rank);
         check(t.yloc-t.ghost_ymin==std::max(0,t.yloc-1), "top halo", rank);
         check(t.xloc+t.width+t.ghost_xmax==std::min(w,t.xloc+t.width+1), "right halo", rank);
         check(t.yloc+t.height+t.ghost_ymax==std::min(h,t.yloc+t.height+1), "bottom halo", rank);
         if (small)
            for (int y=t.yloc; y<t.yloc+t.height; ++y)
               for (int x=t.xloc; x<t.xloc+t.width; ++x) {
                  if (x<0 || x>=w || y<0 || y>=h) MPI_Abort(MPI_COMM_WORLD, 2);
                  ++coverage[y*w+x];
               }
      }
   for (int count : owners) check(count==1, "one tile per rank", rank);
   for (int count : coverage) check(count==1, "each pixel owned once", rank);

   MPI_Barrier(MPI_COMM_WORLD);
   scatterAllTiles(rank, tiles, as.input_data_floats.data(), w, h);
   MPI_Barrier(MPI_COMM_WORLD);
   for (const auto &row : tiles)
      for (const auto &t : row) {
         size_t expected=t.tileRank==rank ?
            (t.width+t.ghost_xmin+t.ghost_xmax)*(t.height+t.ghost_ymin+t.ghost_ymax) : 0;
         check(t.inputBuffer.size()==expected, "local input allocation", rank);
         check(t.outputBuffer.size()==expected, "local output allocation", rank);
      }
   sobelAllTiles(rank, tiles);
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank==0) as.output_data_floats.assign(w*h, -9876.0f);
   MPI_Barrier(MPI_COMM_WORLD);
   gatherAllTiles(rank, tiles, as.output_data_floats.data(), w, h);
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank==0) {
      double maximum_error=0;
      for (int y=0; y<h; ++y)
         for (int x=0; x<w; ++x) {
            float value=as.output_data_floats[y*w+x];
            double error=std::abs(double(value)-reference[y*w+x]);
            check(std::isfinite(value) && error<=1e-6, "assembled float/reference", rank);
            if (x==0 || y==0 || x==w-1 || y==h-1)
               check(value==0, "true global border is zero", rank);
            maximum_error=std::max(maximum_error,error);
         }
      printf("CASE %s P=%d g=%d size=%dx%d max_abs_error=%.9g failures=%d\n",
             label, as.nranks, as.decomp, w, h, maximum_error, failures);
   }
}

void small_cases(AppState &as)
{
   const int sizes[][2]={{13,11},{9,9}};
   const char *labels[]={"constant","ramp","step","irregular"};
   for (const auto &size : sizes)
      for (int pattern=0; pattern<4; ++pattern) {
         int w=size[0], h=size[1];
         as.global_mesh_size[0]=w;
         as.global_mesh_size[1]=h;
         vector<float> reference;
         if (as.myrank==0) {
            as.input_data_floats.resize(w*h);
            reference.resize(w*h);
            for (int y=0; y<h; ++y)
               for (int x=0; x<w; ++x) {
                  float value=0.25f;
                  if (pattern==1) value=x/64.0f;
                  if (pattern==2) value=x<w/2 ? 0.0f : 0.25f;
                  if (pattern==3) value=((x*17+y*23+x*y*3)%256)/256.0f;
                  as.input_data_floats[y*w+x]=value;
               }
            for (int y=0; y<h; ++y)
               for (int x=0; x<w; ++x) {
                  reference[y*w+x]=reference_pixel(as.input_data_floats,w,h,x,y);
                  if (pattern==1 && x>0 && x<w-1 && y>0 && y<h-1)
                     check(reference[y*w+x]==0.125f, "hand-calculated ramp", as.myrank);
               }
         }
         for (int decomp=1; decomp<=3; ++decomp) {
            as.decomp=decomp;
            distributed_case(as,reference,labels[pattern],true);
         }
      }
}

void image_cases(AppState &as, const char *input, int w, int h, const char *output)
{
   if (strlen(input)>=sizeof(as.input_filename) || strlen(output)>=sizeof(as.output_filename)
       || w<as.nranks || h<as.nranks) {
      fprintf(stderr,"Image dimensions/path length unsupported by this check.\n");
      MPI_Abort(MPI_COMM_WORLD,2);
   }
   as.global_mesh_size[0]=w;
   as.global_mesh_size[1]=h;
   strcpy(as.input_filename,input);
   strcpy(as.output_filename,output);
   vector<float> reference;
   if (as.myrank==0) {
      std::ifstream file(input,std::ios::binary|std::ios::ate);
      if (!file || file.tellg()!=std::streamoff(w)*h) {
         fprintf(stderr,"Input byte count does not match width*height.\n");
         MPI_Abort(MPI_COMM_WORLD,2);
      }
      loadInputFile(&as);
      reference.resize(w*h);
      // Full-domain graded Sobel: small_cases independently checks the math.
      do_sobel_filtering(as.input_data_floats.data(),reference.data(),w,h);
      as.output_data_floats=reference;
      writeOutputFile(as); // Same existing byte conversion as the production program.
      size_t above_one=0;
      for (float value : reference) if (value>1.0f) ++above_one;
      printf("REFERENCE floats_above_one=%zu; byte comparison checks existing conversion parity only.\n",above_one);
   }
   for (int decomp=1; decomp<=3; ++decomp) {
      as.decomp=decomp;
      distributed_case(as,reference,"image",false);
   }
}
} // namespace

int main(int argc, char **argv)
{
   MPI_Init(&argc,&argv);
   AppState as;
   MPI_Comm_rank(MPI_COMM_WORLD,&as.myrank);
   MPI_Comm_size(MPI_COMM_WORLD,&as.nranks);
   char hostname[256];
   gethostname(hostname,sizeof(hostname));
   printf("Validation rank %d of %d on <%s>\n",as.myrank,as.nranks,hostname);
   fflush(stdout);
   if (as.myrank==0) {
      char version[MPI_MAX_LIBRARY_VERSION_STRING];
      int length;
      MPI_Get_library_version(version,&length);
      printf("MPI library: %.*s\n",length,version);
   }
   bool transfer=argc==2 && strcmp(argv[1],"transfers")==0 && as.nranks==2;
   bool small=argc==2 && strcmp(argv[1],"small")==0 && (as.nranks==4 || as.nranks==9);
   bool image=argc==6 && strcmp(argv[1],"image")==0 && (as.nranks==4 || as.nranks==9);
   if (transfer) transfers(as.myrank);
   else if (small) small_cases(as);
   else if (image) image_cases(as,argv[2],atoi(argv[3]),atoi(argv[4]),argv[5]);
   else {
      if (as.myrank==0) fprintf(stderr,"Usage: 2 ranks: transfers; 4/9 ranks: small | image INPUT W H REFERENCE_OUTPUT\n");
      MPI_Abort(MPI_COMM_WORLD,2);
   }
   int total_failures=0;
   MPI_Allreduce(&failures,&total_failures,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   if (as.myrank==0)
      printf("%s: MPI validation %s; failures=%d\n",total_failures ? "FAIL" : "PASS",argv[1],total_failures);
   MPI_Finalize();
   return total_failures ? 1 : 0;
}
