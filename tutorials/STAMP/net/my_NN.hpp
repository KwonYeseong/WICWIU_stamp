#include <iostream>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

class my_NN : public NeuralNetwork<float>{
private:
public:
    my_NN(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = new Linear<float>(out, 900, 320, TRUE, "1");
        out = new Tanh<float>(out, "Tanh");

        out = new Linear<float>(out, 320, 15, TRUE, "2");
        out = new Tanh<float>(out, "Tanh");
        // ======================= layer 2=======================
        out = new Linear<float>(out, 15, 2, TRUE, "3");

        AnalyzeGraph(out);


        // ======================= Select LossFunction Function ===================
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.0001, 0.9, MINIMIZE));

    }

    virtual ~my_NN() {}
};
