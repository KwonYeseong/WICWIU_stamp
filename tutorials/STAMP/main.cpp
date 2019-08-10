#include "net/my_CNN.hpp"
#include "net/my_NN.hpp"
#include "BMP_Reader.hpp"
#include <time.h>

#define BATCH             32
#define EPOCH             50
#define LOOP_FOR_TRAIN    (1558 / BATCH)
#define LOOP_FOR_TEST     (667 / BATCH)
#define NUMBER_OF_CLASS 2

int main(int argc, char const *argv[]) {
    clock_t startTime,  endTime;
    double  nProcessExcuteTime;
    char filename[]      = "./STAMP_parmas";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *x    = new Tensorholder<float>(1, BATCH, 1, 1, 900, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 2, "label");

    // ======================= Select net ===================
    NeuralNetwork<float> *net = new my_CNN(x, label);
    // NeuralNetwork<float> *net = new my_NN(x, label);
    // ======================= Prepare Data ===================
    BMPDataSet<float> *dataset = CreatBMPDataSet<float>(BATCH);


    net->PrintGraphInformation();

    float best_acc = 0;
    int   epoch    = 0;
    Tensor<float> **data = NULL;

    // @ When load parameters
    // net->Load(filename);

    std::cout << "best_acc : " << best_acc << '\n';
    std::cout << "epoch : " << epoch << '\n';

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        dataset->prepareData();

        // ======================= Train =======================
        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();
        
        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            data = dataset->GetTrainDataFromBuffer();
            
            net->FeedInputTensor(2, data[0], data[1]);
            delete data;
            data = NULL;
            net->ResetParameterGradient();
            net->Train();

            train_accuracy += net->GetAccuracy(NUMBER_OF_CLASS);
            train_avg_loss += net->GetLoss();

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
                   j + 1, LOOP_FOR_TRAIN,
                   train_avg_loss / (j + 1),
                   train_accuracy / (j + 1)
                   /*nProcessExcuteTime*/);
            fflush(stdout);
        }
        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            data = dataset->GetTestDataFromBuffer();
            
            net->FeedInputTensor(2, data[0], data[1]);
            delete data;
            data = NULL;
            net->Test();

            test_accuracy += net->GetAccuracy(NUMBER_OF_CLASS);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, LOOP_FOR_TEST,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << "\n\n";

        // if ((best_acc < (test_accuracy / LOOP_FOR_TEST))) {
        //     net->Save(filename);
        // }
    }

    delete dataset;
    delete net;

    return 0;
}
