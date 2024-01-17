#include <memory>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <condition_variable>
#include <queue>
#include <barrier>
#include <latch>
#include <concepts>


unsigned get_num_thread();
void set_num_threads(unsigned T);

struct partial_sum_t
{
    alignas(64) double value;
};

typedef struct profiling_results_t
{
    double result, time, speedup, efficiency;
    unsigned T;
} profiling_results_t;

template <class F>
auto run_experiment(F func, const double *v, size_t n)
requires std::is_invocable_r_v<double, F, const double *, size_t>
{
    std::vector<profiling_results_t> res_table;
    auto Tmax = get_num_thread(); 
    for (unsigned int T = 1; T <= Tmax; ++T)
    {
        using namespace std::chrono;
        res_table.emplace_back();
        auto& rr = res_table.back();
        set_num_threads(T);
        auto t1 = steady_clock::now();
        rr.result = func(v, n);
        auto t2 = steady_clock::now();
        rr.time = duration_cast<milliseconds>(t2 - t1).count();
        rr.speedup = res_table.front().time / rr.time;
        rr.efficiency = rr.speedup / T;
        rr.T = T;
    }
    return res_table;
}

double average(const double *v, size_t n)
{
    double res = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        res += v[i];
    }
    return res / n;
}

double average_reduce(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel for reduction(+ : res)
    for (int i = 0; i < n; ++i)
    {
        res += v[i];
    }
    return res / n;
}

double average_rr(const double *v, size_t n) // Roll Round
{
    double res = 0.0;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        for (int i = t; i < n; i += T)
        {
            res += v[i]; // Гонка
        }
    }
    return res / n;
}

double average_omp(const double *v, size_t n)
{
    double res = 0.0, *partial_sums;
    unsigned T;
#pragma omp parallel shared(T)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (double *)calloc(T, sizeof(v[0]));
        }
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t] += v[i];
        }
    }
    for (size_t i = 1; i < omp_get_num_procs(); ++i)
    {
        partial_sums[0] += partial_sums[i];
    }
    res = partial_sums[0] / n;
    free(partial_sums);
    return res;
}

double average_omp_align(const double *v, size_t n)
{
    double res = 0.0;
    partial_sum_t *partial_sums;
    unsigned T;
#pragma omp parallel shared(T)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t *)calloc(T, sizeof(partial_sum_t));
        }
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t].value += v[i];
        }
    }
    for (size_t i = 1; i < T; ++i)
    {
        partial_sums[0].value += partial_sums[i].value;
    }
    res = partial_sums[0].value / n;
    free(partial_sums);
    return res;
}

double average_omp_mtx(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel
    {
        unsigned int t = omp_get_thread_num();
        unsigned int T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {
         #pragma omp critical
            {
                res += v[i];
            }
        }
    }
    return res / n;
}

double average_omp_mtx_opt(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel
    {
        double partial = 0.0;
        unsigned int t = omp_get_thread_num();
        unsigned int T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {

            partial += v[i];
        }
#pragma omp critical
        {
            res += partial;
        }
    }
    return res / n;
}

double average_cpp_mtx(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    std::mutex mtx;
    auto worker_proc = [&mtx, T, n, &res, v](unsigned t)
    {
        double partial_result = 0.0;
        for (std::size_t i = t; i < n; i += T)
        {
            partial_result += v[i];
        }
        std::scoped_lock l{mtx};
        res += partial_result;
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    return res / n;
}

double average_cpp_partial_align(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    partial_sum_t *partial_sums = (partial_sum_t *)calloc(T, sizeof(partial_sum_t));
    auto worker_proc = [v, n, T, &res, partial_sums](size_t t)
    {
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t].value += v[i];
        }
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    for (size_t i = 1; i < T; ++i)
    {
        partial_sums[0].value += partial_sums[i].value;
    }
    res = partial_sums[0].value / n;
    free(partial_sums);
    return res;
}

double average_mtx_local(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = get_num_thread();
    std::mutex mtx;
    size_t e = n / T;
    size_t b = n % T;
    std::vector<std::thread> workers;
    auto worker_proc = [v, n, T, &res, &mtx](size_t t, size_t e, size_t b)
    {
        double local = 0.0;
        if (t < b)
        {
            b = t * ++e;
        }
        else
        {
            b += t * e;
        }
        e += b;
        for (size_t i = b; i < e; ++i)
        {
            local += v[i];
        }
        std::scoped_lock l{mtx};
        res += local;
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t, e, b);
    }
    worker_proc(0, e, b);
    for (auto &w : workers)
    {
        w.join();
    }
    return res / n;
}

int producers_consumers() {
    unsigned p = get_num_thread();
    unsigned producers = p/2, consumers = p-producers;
    
    std::queue<int> q;
    std::vector<std::thread> vProducers;
    std::vector<std::thread> vConsumers;
    std::mutex mtx;
    std::condition_variable cv;
    auto producer_work = [&q, &mtx, &cv](int n) {
        
        for (int i = 0; i < n; i++) {
            std::scoped_lock lock(mtx);
            q.push(i);
            cv.notify_one();
        }
    };
    auto consumer_work = [&q, &mtx, &cv](unsigned t) {
        std::unique_lock ul(mtx);
        while (q.empty()) { cv.wait(ul); };
        int message = q.front();
        q.pop();
        std::cout << "Thread " << t << " received message " << message << '\n';
        ul.unlock();
    };
    for (unsigned t = 1; t <= producers; ++t)
    {
        vProducers.emplace_back(producer_work, 100);
    }
    for (unsigned t = 1; t <= consumers; ++t)
    {
        vConsumers.emplace_back(consumer_work, t);
    }
    for (auto& w : vProducers)
    {
        w.join();
    }
    for (auto& w : vConsumers)
    {
        w.join();
    }
    return 0;
}

//Барьеры
//1разовый - latch
// latch(T);
// arrive_and_wait();
// многоразовый barrier
// barrier(T)
// arrive_and_wait();


//parallel reduction

double average_cpp_reduction(const double* v, const size_t& n) {
    unsigned T = get_num_thread();

    std::vector<double> partial_results(T);
    for (auto& elem : partial_results)
        elem = 0;

    size_t e = n / T;
    size_t b = n % T;

    barrier bar(T);

    auto fillPart = [&partial_results, &v]( size_t e, size_t b, const size_t& t){

        double local = 0.0;
        if (t < b)
        {
            b = t * ++e;
        }
        else
        {
            b += t * e;
        }
        e += b;
        for (size_t i = b; i < e; ++i)
        {
            local += v[i];
        }
        partial_results[t] = local;
    };

    auto work = [&partial_results, &T, &bar, &v, &fillPart, &e, &b](int t) {
        fillPart(e,b,t);
        for (size_t step = 1, next = 2; step < T; step = next, next += next) {
            bar.arrive_and_wait();
            if ((t & (next - 1)) == 0 && t+step<T) {
                partial_results[t] += partial_results[t+step];
            }
        }
    };
    std::vector<std::thread> workers;
    for (size_t i = 0; i < T; i++)
        workers.emplace_back(work, i);
    for (auto& worker : workers)
        worker.join();


    return partial_results[0]/n;
}


uint32_t myMax(const uint32_t& a, const uint32_t& b) {
    if (a > b)
        return a;
    return b;
}

double randomize_vector( uint32_t* v, size_t n, uint32_t seed, uint32_t minVal=0, uint32_t maxVal=UINT32_MAX) {
    if (minVal > maxVal)
        exit(__LINE__);
    auto A = 22695477;
    auto B = 1;
    unsigned long M = 1 << 32;
    double res = 0.0;
    uint32_t mod = maxVal - minVal + 1;
    if(mod==0)
        for (size_t i = 0; i < n; i++) {
            seed = seed * A + B;
            v[i] = seed;
            res += v[i];
        }
    else
        for (size_t i = 0; i < n; i++) {
            seed = seed * A + B;
            v[i] = minVal+(seed%(maxVal-minVal+1));
            res += v[i];
        }
    return res/n;
}

class bc_t {
    uint32_t A, B;
public:
    bc_t(uint32_t a, uint32_t b) : A(a), B(b) {

    }

    bc_t& operator*=(const bc_t& x) {
        A *= x.A;
        B += A * x.B;
        return *this;
    }
    auto operator()(uint32_t seed) const {
        return A * seed + B;
    }
};

int main()
{
    //size_t N = 1u << 25;
    //double v, t1, t2;

    //char pattern[] = "%f, %f %s\n";

    //auto buf = std::make_unique<double[]>(N);
    //for (size_t i = 0; i < N; ++i)
    //    buf[i] = i;

    //t1 = omp_get_wtime();
    //v = average(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "sequential");

    //t1 = omp_get_wtime();
    //v = average_reduce(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "reduce");

    //t1 = omp_get_wtime();
    //v = average_rr(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "rr");

    //t1 = omp_get_wtime();
    //v = average_omp(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "omp");

    //t1 = omp_get_wtime();
    //v = average_omp_align(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "omp_align");

    //t1 = omp_get_wtime();
    //v = average_omp_mtx(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "omp_mtx");

    //t1 = omp_get_wtime();
    //v = average_omp_mtx_opt(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "omp_mtx_opt");

    //t1 = omp_get_wtime();
    //v = average_cpp_mtx(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "cpp_mtx");

    //t1 = omp_get_wtime();
    //v = average_cpp_partial_align(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "cpp_partial_align");

    //t1 = omp_get_wtime();
    //v = average_mtx_local(buf.get(), N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "cpp_mtx_local");


    ////auto a = run_experiment(average_mtx_local, buf.get(), N);
    //t1 = omp_get_wtime();
    //v = average_cpp_reduction(buf.get(),N);
    //t2 = omp_get_wtime();
    //printf(pattern, v, t2 - t1, "average_cpp_reduce");
    size_t n = 20;
    uint32_t* vec = new uint32_t[n];
    std::cout << randomize_vector(vec, n,501) << "\n";
    for (size_t i = 0; i < n; i++)
        std::cout << vec[i] << " ";
    return 0;
}
