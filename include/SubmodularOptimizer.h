#ifndef SUBMODULAROPTIMIZER_H
#define SUBMODULAROPTIMIZER_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>
#include <memory>

#include "SubmodularFunction.h"

class SubmodularFunctionWrapper : public SubmodularFunction {
protected:
    std::function<data_t (std::vector<std::vector<data_t>> const &)> f;

public:

    SubmodularFunctionWrapper(std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : f(f) {
    }

    std::shared_ptr<SubmodularFunction> clone() const {
        return std::shared_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f));
    }

    data_t operator()(std::vector<std::vector<data_t>> const &solution) const {
        return f(solution);
    }
};

class SubmodularOptimizer {
private:
    
protected:
    unsigned int K;
    
    // THIS REQUIRES OUR OWN COPY CTOR IN THIS CLASS
    // SubmodularFunction * wrapper = NULL;
    // SubmodularFunction &f;
    
    /** 
     * Okay lets explain the reasoning behind this a bit more. 
     * 1):  Use SubmodularFunction as an object. This does not work because we are dealing with inheritance / virtual functions
     * 2):  Use a reference to SubmodularFunction. This works well with PyBind and has an easy interface, but breaks does not work well 
     *      together with the SubmodularFunctionWrapper if we want to allow users to pass a std::function directly.
     * 3):  Use SubmodularFunction* which would be a very "pure" and old-school approach. Should be do-able, but is not the modern c++ style
     * 4):  Use std::unique_ptr<SubmodularFunction>. This would IMHO be the best approach as it reflects our intend, that the SubmodularOptimizer owns the SubmodularFunction which it clones beforehand. This does not work well with PyBind since PyBind wants to own the memory. 
     * 5):  Use std::shared_ptr<SubmodularFunction> which is basically a modern version of 3) and works better with PyBind. 
     *
     **/
    //std::unique_ptr<SubmodularFunction> f;
    std::shared_ptr<SubmodularFunction> f;

    //std::function<data_t (std::vector<std::vector<data_t>> const &)> f;
    bool is_fitted;

public:
    //TODO: Do we want to have this public here?
    std::vector<std::vector<data_t>> solution;
    data_t fval;

    // SubmodularOptimizer(unsigned int K, std::unique_ptr<SubmodularFunction> f) : K(K), f(std::move(f)) {
    // }

    SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
        : K(K), f(f.clone()) {
        is_fitted = false;
        fval = 0;
        assert(("K should at-least be 1 or greater.", K >= 1));
    }
    
    // SubmodularOptimizer(unsigned int K, SubmodularFunction * f) 
    //     : SubmodularOptimizer(K, std::move(std::unique_ptr<SubmodularFunction>(f))) {}

    SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
        : K(K), f(std::unique_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f))) {
        is_fitted = false;
        fval = 0;
        assert(("K should at-least be 1 or greater.", K >= 1));
    }

    //  SubmodularOptimizer(unsigned int K, data_t (std::vector<std::vector<data_t>> const &) f) 
    //     : SubmodularOptimizer(K, [f](std::vector<std::vector<data_t>> const &X){return f(X);}) {}

    // SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
    //     : K(K), f(f) {
    //     is_fitted = false;
    //     assert(("K should at-least be 1 or greater.", K >= 1));
    // }

    // SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
    //     : K(K), wrapper(new SubmodularFunctionWrapper(f)), f(*wrapper) {}

    /**
     *
     * @param dataset
     * @return
     */
    virtual void fit(std::vector<std::vector<data_t>> const & X) = 0;

    /**
     *
     * @param dataset
     * @return
     */
    virtual void next(std::vector<data_t> const &x) = 0;

    /**
     *
     * @param dataset
     * @return
     */
    std::vector<std::vector<data_t>>const &  get_solution() const {
        if (!this->is_fitted) {
             throw std::runtime_error("Optimizer was not fitted yet! Please call fit() or next() before calling get_solution()");
        } else {
            return solution;
        }
    }
    
    data_t get_fval() const {
        return fval;
    }

    /**
     * Destructor.
     */
    virtual ~SubmodularOptimizer() {
        // if (wrapper != NULL) {
        //     delete wrapper;
        //     wrapper = NULL;
        // }
    }
};

#endif // THREESIEVES_SUBMODULAROPTIMIZER_H
