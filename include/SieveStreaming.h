#ifndef SIEVESTREAMING_H
#define SIEVESTREAMING_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

class SieveStreaming : public SubmodularOptimizer {
protected:

    class Sieve : public SubmodularOptimizer {
    public:
        data_t threshold;

        Sieve(unsigned int K, SubmodularFunction & f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}

        Sieve(unsigned int K, std::shared_ptr<SubmodularFunction> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {}

        Sieve(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t threshold) : SubmodularOptimizer(K,f), threshold(threshold) {
        }

        void fit(std::vector<std::vector<data_t>> const & X) {
            throw std::runtime_error("Sieves are only meant to be used through SieveStreaming and therefore do not require the implementation of `fit'");
        }

        void next(std::vector<data_t> const &x) {
            unsigned int Kcur = solution.size();
            if (Kcur < K) {
                data_t fdelta = f->peek(solution, x, solution.size()) - fval;
                data_t tau = (threshold / 2.0 - fval) / static_cast<data_t>(K - Kcur);
                if (fdelta >= tau) {
                    f->update(solution, x, solution.size());
                    solution.push_back(std::vector<data_t>(x));
                    fval += fdelta;
                }
            }
        }

    };

    static inline std::vector<data_t> thresholds(data_t lower, data_t upper, data_t epsilon) {
        if (epsilon > 0.0) {
            // unsigned int i = 0;
            // do {
            //     data_t val = std::pow(1+epsilon, i);
            //     if (val > lower) {
            //         thresholds.push_back(val);
            //     }
            // } while( val < upper);

            int ilower = std::ceil(std::log(lower) / std::log(1.0 + epsilon));
            int iupper; // = std::floor(std::log(upper) / std::log(1.0 + epsilon));

            data_t tmp = std::log(upper) / std::log(1.0 + epsilon);
            if (tmp == std::floor(tmp)) {
                iupper = std::floor(tmp) - 1;
            } else {
                iupper = std::floor(tmp);
            }

            if (ilower >= iupper)
                throw std::runtime_error("SieveStreaming::thresholds: Lower threshold boundary (" + std::to_string(ilower) + ") is higher than or equal to the upper boundary ("
                                        + std::to_string(iupper) + "), epsilon = " + std::to_string(epsilon) + ".");

            std::vector<data_t> ts;
            for (int i = ilower; i < iupper; ++i) {
                ts.push_back(std::pow(1.0 + epsilon, i));
            }

            return ts;
        } else
            throw std::runtime_error("SieveStreaming::thresholds: epsilon must be a positive real-number (is: " + std::to_string(epsilon) + ").");
    }

protected:
    std::vector<Sieve*> sieves;

public:
    SieveStreaming(unsigned int K, SubmodularFunction & f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);
        for (auto t : ts) {
            sieves.push_back(new Sieve(K, f, t));
        }
    }

    SieveStreaming(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f, data_t m, data_t epsilon) : SubmodularOptimizer(K,f) {
        std::vector<data_t> ts = thresholds(m, K*m, epsilon);
        for (auto t : ts) {
            sieves.push_back(new Sieve(K, f, t));
        }
    }

    ~SieveStreaming() {
        for (auto s : sieves) {
            delete s;
        }
    }

    void fit(std::vector<std::vector<data_t>> const & X) {
        for (auto &x : X) {
            next(x);
        }
    }

    void next(std::vector<data_t> const &x) {
        for (auto s : sieves) {
            s->next(x);
            if (s->get_fval() > fval) {
                fval = s->get_fval();
                // TODO THIS IS A COPY AT THE MOMENT
                solution = s->solution;
            }
        }
        is_fitted = true;
    }
};

#endif