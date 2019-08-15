#include "net/my_CNN.hpp"
#include "net/my_NN.hpp"
#include "../../WICWIU_src/Tensor.hpp"

#include "string.h"

class HGUStampRecog{
private:
    NeuralNetwork<float> *net;
    Tensorholder<float> *x;
    Tensorholder<float> *label;
    
public:
    HGUStampRecog(){
        NeuralNetwork<float> *net = NULL;
        Tensorholder<float> *x = NULL;
        Tensorholder<float> *label = NULL;
    }
    
    HGUStampRecog(char *modelFile){
        net = NULL;
        x = new Tensorholder<float>(1, 1, 1, 1, 900, "x");
        label = new Tensorholder<float>(1, 1, 1, 1, 2, "dump");
        
        Load(modelFile);
    }
    
    ~HGUStampRecog(){
        Delete();
    }
    
    // create model and load parameters
    int Load(char *modelFile){
        net = new my_CNN(x, label);
        net->Load(modelFile);
        std::cout << "loaded pretrained model" << std::endl;
    }
    
    // Recognize an input image return TRUE(1) or FALSE(0)
    int RecognizeStamp(int width, int height, unsigned char *gray){
        Tensor<float> *input = Tensor<float>::Zeros(1, 1, 1, width, height);
        Tensor<float> *dummy = Tensor<float>::Zeros(1, 1, 1, 1, 2);
        Operator<float> *resultOp = NULL;
        int result = -1;
        
        input = BMP2Tensor(width, height, gray);
        
        net->FeedInputTensor(2, input, dummy);
        net->Test();
        resultOp = net->GetResultOperator();
        Tensor<float> *pred = resultOp->GetResult();
        
        // printf("%f %f\n", (*pred)[0], (*pred)[1]);
        
        if((*pred)[0] > (*pred)[1])
        return 0;
        else
        return 1;
    }
    
    Tensor<float> *BMP2Tensor(int width, int height, unsigned char *gray){
        Tensor<float> *temp = Tensor<float>::Zeros(1, 1, 1, width, height);
        
        for (int i = 0; i < height ; i++){
            for (int j = 0; j < width; j++){
                (*temp)[Index5D(temp->GetShape(), 0, 0, 0, j, i)] = gray[i*width + j] / 255.0;
            }
        }
        
        return temp;
    }
    
    // deallocate model (distructor)
    int Delete(){
        delete net;
        delete x;
        delete label;
    }
};
