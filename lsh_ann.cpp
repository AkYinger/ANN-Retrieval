// lsh_ann.cpp
// Approximate Nearest Neighbor search using LSH (random hyperplane sign hashing)
// for 128‑dimensional vectors. Provides sequential and OpenMP‑parallel query paths.
// 
// Dataset: using any *.fvecs set from http://corpus-texmex.irisa.fr
// 
// Compile:
//   g++ -O3 -march=native -std=c++17 -fopenmp lsh_ann.cpp -o lsh_ann
//
// Run:
//   lsh_ann sift_base.fvecs sift_query.fvecs 12 16 10 6
//   # 12 hash tables, 16 hyperplanes / table, return 10 neighbours, 6 threads
//
// --------------------
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

//  LSH index 
class LSHIndex {
public:
    LSHIndex(int dim, int L, int k);
    void build(const vector<vector<float>>& base);
    vector<int> query(const vector<float>& q, int topk) const;

private:
    int dim_, L_, k_;
    vector<vector<vector<float>>> hyperplanes_;        // [L][k][dim]
    vector<unordered_map<uint64_t, vector<int>>> tbl_; // [L] hash→ids
    const vector<vector<float>>* data_ = nullptr;      // pointer to base data

    uint64_t hash_vec(const vector<float>& v, int table) const;
    inline float dot(const vector<float>& a, const vector<float>& b) const;
};

LSHIndex::LSHIndex(int dim, int L, int k)
    : dim_(dim), L_(L), k_(k), hyperplanes_(L, vector<vector<float>>(k, vector<float>(dim))),
      tbl_(L) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int t = 0; t < L_; ++t)
        for (int j = 0; j < k_; ++j)
            for (int d = 0; d < dim_; ++d)
                hyperplanes_[t][j][d] = nd(rng);
}

inline float LSHIndex::dot(const vector<float>& a, const vector<float>& b) const {
    float s = 0.f;
    for (int i = 0; i < dim_; ++i) s += a[i] * b[i];
    return s;
}

uint64_t LSHIndex::hash_vec(const vector<float>& v, int table) const {
    uint64_t h = 0;
    for (int j = 0; j < k_; ++j)
        if (dot(v, hyperplanes_[table][j]) > 0)
            h |= (1ULL << j);
    return h;
}

void LSHIndex::build(const vector<vector<float>>& base) {
    data_ = &base;
    for (int id = 0; id < static_cast<int>(base.size()); ++id)
        for (int t = 0; t < L_; ++t)
            tbl_[t][hash_vec(base[id], t)].push_back(id);
}

vector<int> LSHIndex::query(const vector<float>& q, int topk) const {
    unordered_set<int> cand;
    for (int t = 0; t < L_; ++t) {
        auto it = tbl_[t].find(hash_vec(q, t));
        if (it != tbl_[t].end()) cand.insert(it->second.begin(), it->second.end());
    }
    struct Node { int id; float dist; };
    auto cmp = [](const Node& a, const Node& b) { return a.dist < b.dist; };
    priority_queue<Node, vector<Node>, decltype(cmp)> pq(cmp);

    for (int id : cand) {
        const auto& v = (*data_)[id];
        float d = 0.f;
        for (int i = 0; i < dim_; ++i) {
            float diff = v[i] - q[i];
            d += diff * diff;
        }
        if (static_cast<int>(pq.size()) < topk) pq.push({id, d});
        else if (d < pq.top().dist) { pq.pop(); pq.push({id, d}); }
    }

    vector<int> res(pq.size());
    for (int i = static_cast<int>(res.size()) - 1; i >= 0; --i) { res[i] = pq.top().id; pq.pop(); }
    return res;
}

//  I/O helpers 
vector<vector<float>> read_fvecs(const string& path) {
    ifstream fin(path, ios::binary);
    if (!fin) throw runtime_error("Cannot open " + path);
    uint32_t dim; fin.read(reinterpret_cast<char*>(&dim), 4);
    fin.seekg(0);
    vector<vector<float>> vecs;
    while (true) {
        uint32_t d; fin.read(reinterpret_cast<char*>(&d), 4);
        if (fin.eof()) break;
        if (d != dim) throw runtime_error("Non‑uniform vector size in " + path);
        vector<float> v(d); fin.read(reinterpret_cast<char*>(v.data()), d * sizeof(float));
        vecs.emplace_back(move(v));
    }
    return vecs;
}

//  main / benchmark 
int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " base.fvecs query.fvecs L k topK [threads]\n";
        return 1;
    }
    string base_path  = argv[1];
    string query_path = argv[2];
    int L     = atoi(argv[3]);    // hash tables
    int k     = atoi(argv[4]);    // hyperplanes per table
    int topK  = atoi(argv[5]);    // neighbours to return
    int nthr  = (argc >= 7) ? atoi(argv[6]) : omp_get_max_threads();

    auto base   = read_fvecs(base_path);
    auto querys = read_fvecs(query_path);
    int dim = base.empty() ? 0 : base[0].size();
    cout << "Loaded " << base.size() << " base, " << querys.size() << " queries, dim=" << dim << "\n";

    LSHIndex index(dim, L, k);
    auto t0 = chrono::steady_clock::now();
    index.build(base);
    auto t1 = chrono::steady_clock::now();
    cout << "Index build: "
         << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms\n";

    // Sequential
    vector<vector<int>> ans_seq(querys.size());
    t0 = chrono::steady_clock::now();
    for (size_t i = 0; i < querys.size(); ++i) ans_seq[i] = index.query(querys[i], topK);
    t1 = chrono::steady_clock::now();
    double seq_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cout << "Sequential: " << seq_ms << " ms total, " << seq_ms / querys.size() << " ms/q\n";

    // Parallel
    omp_set_num_threads(nthr);
    vector<vector<int>> ans_par(querys.size());
    t0 = chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < querys.size(); ++i) ans_par[i] = index.query(querys[i], topK);
    t1 = chrono::steady_clock::now();
    double par_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cout << "Parallel (" << nthr << " th): " << par_ms << " ms total, " << par_ms / querys.size() << " ms/q\n";

    return 0;
}
