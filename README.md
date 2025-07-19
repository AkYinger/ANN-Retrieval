OVERVIEW
This project consists of a single file, lsh_ann.cpp.  
It implements:  
• Random‑hyperplane LSH for 128‑D vectors  
• An index builder and query routine (LSHIndex)  
• Sequential and OpenMP‑parallel batch‑query loops  
• Simple timing output  
  
PREREQUISITES  
• C++17 compiler (tested: g++ 13.2)  
• OpenMP 4.5 or newer (included with recent g++)  
• *.fvecs datasets from http://corpus-texmex.irisa.fr  
  
BUILD  
g++ -O3 -march=native -std=c++17 -fopenmp lsh_ann.cpp -o lsh_ann  
(omit “‑fopenmp” if you only need the sequential path)  
  
RUN  
./lsh_ann sift_base.fvecs sift_query.fvecs 12 16 10 6  
^base file ^queries file ^L ^k ^topK ^threads  
  
PARAMETER GUIDE  
• L – number of hash tables (8–16 is common)  
• k – bits / hyperplanes per table (12–20)  
• topK – neighbours returned (10–100)  
• threads – OpenMP threads (no more than physical cores recommended)  
  
DIRECTORY LAYOUT  
.  
├── lsh_ann.cpp  
├── README.md   
└── (place *.fvecs files here)  
