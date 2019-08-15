#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

class my_CNN : public NeuralNetwork<float>{
private:
public:
    my_CNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        SetInput(2, x, label);

        Operator<float> *out = NULL;

        out = new ReShape<float>(x, 30, 30, "Flat2Image");

        // ======================= layer 1=======================
        out = new ConvolutionLayer2D<float>(out, 1, 32, 4, 4, 2, 2, 0, FALSE, "Conv_1");
        out = new Relu<float>(out, "Relu_1");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 32, 64, 4, 4, 2, 2, 0, FALSE, "Conv_2");
        out = new Relu<float>(out, "Relu_2");

        // ======================= layer 2=======================
        out = new ConvolutionLayer2D<float>(out, 64, 64, 4, 4, 2, 2, 0, FALSE, "Conv_3");
        out = new Relu<float>(out, "Relu_3");

        // ======================= layer 3=======================
        out = new ReShape<float>(out, 1, 1, 2 * 2 * 64, "Image2Flat");

        // ======================= layer 3=======================
        out = new Linear<float>(out, 2 * 2 * 64, 500, TRUE, "Fully-Connected_1");
        out = new Relu<float>(out, "Relu_4");
        
        //// ======================= layer 4=======================
        out = new Linear<float>(out, 500, 2, TRUE, "Fully-connected_2");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.003, 0.9, MINIMIZE));
        // SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        // SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        // SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
    }

    virtual ~my_CNN() {}
};
