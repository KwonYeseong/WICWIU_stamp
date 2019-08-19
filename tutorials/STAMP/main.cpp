// /* This code for Test */
// #include <string.h>
// #include <stdio.h>
// #include <stdlib.h>

// #include <iostream>

// #include "./HGUStampRecog.h"

// #define BYTE unsigned char
// #define DWORD int
// #define WORD unsigned short
// #define LONG int

// #pragma pack(push,1)

// typedef struct tagBITMAPFILEHEADER {
//     short bfType;
//     int   bfSize;
//     short bfReserved1;
//     short bfReserved2;
//     int   bfOffBits;
// } BITMAPFILEHEADER;

// typedef struct tagBITMAPINFOHEADER{
//     DWORD biSize;          //현 구조체의 크기
//     LONG biWidth;          //이미지의 가로 크기
//     LONG biHeight;         //이미지의 세로 크기
//     WORD biPlanes;         //플레인수
//     WORD biBitCount;       //비트 수
//     DWORD biCompression;   //압축 유무
//     DWORD biSizeImage;     //이미지 크기
//     LONG biXPelsPerMeter;  //미터당 가로 픽셀
//     LONG biYPelsPerMeter;  //미터당 세로 픽셀
//     DWORD biClrUsed;       //컬러 사용 유무
//     DWORD biClrImportant;  //중요하게 사용하는 색
// }BITMAPINFOHEADER;

// typedef struct tagRGBQUAD {
//     BYTE   rgbBlue;
//     BYTE   rgbGreen;
//     BYTE   rgbRed;
//     BYTE   rgbReserved;
// }  RGBQUAD;

// #pragma pack(pop)

// int checkFileDir(char* dir){
//     FILE *temp = fopen(dir, "rb");

//     if(temp == NULL){
//         fclose(temp);
//         return 0;
//     }
//     else {
//         fclose(temp);
//         return 1;
//     }
// }

// unsigned char *ReadBMP(char *imageDir, int *width, int *height){
//         FILE *input;
//         unsigned char *bmp;
//         long size;

//         input = fopen(imageDir, "rb");
//         if (input == NULL){
//             printf("Can not open the file \"%s\"\n", imageDir);
//             std::cout << imageDir << std::endl;
//         }
//         fseek(input, 0, SEEK_END);
//         size = ftell(input);
//         rewind(input);

//         bmp = new unsigned char[size];
//         int dummy = fread(bmp, 1, size, input);
//         fclose(input);

//         BITMAPFILEHEADER *bf = (BITMAPFILEHEADER *)bmp;
//         BITMAPINFOHEADER *bi = (BITMAPINFOHEADER *)(bmp + sizeof(*bf));
//         RGBQUAD *gr = (RGBQUAD *)(bmp + sizeof(*bf) + sizeof(*bi));
//         unsigned char *img = bmp + bf->bfOffBits;

//         if (bf->bfType != 0x4D42){
//             printf("Not bitmap file!");
//         }
//         if (bi->biBitCount != 8){
//             printf("Bad file format!");
//         }

//         int w = bi->biWidth;
//         if (w % 4)
//             w += 4 - bi->biWidth % 4;

//         unsigned char *temp = (unsigned char*)malloc(sizeof(unsigned char) * bi->biHeight * bi->biWidth); // 크기를 이미지에 맞출 것
//         unsigned char B, G, R;
//         int p;

//         for (int i = 0; i < bi->biHeight; i++){
//             for (int j = 0; j < bi->biWidth; j++){
//                 p = img[(bi->biHeight - i - 1) * w + j];
//                 B = gr[p].rgbBlue;
//                 G = gr[p].rgbGreen;
//                 R = gr[p].rgbRed;
//                 temp[i * bi->biWidth + j] = (((114 * R + 587 * G + 299 * B) / 1000));
//             }
//         }

//         *width = bi->biWidth;
//         *height = bi->biHeight;

//         free(bmp);
//         return temp;
//     }


// int main(int argc, char const *argv[]){
//     int img_width = 0, img_height = 0;
//     int result = -1;
//     char paramsFileName[] = "./STAMP_parmas";  //default params file
//     char filename[256] = "\0";
//     unsigned char *image = NULL;

//     if(argc > 1){
//         strcpy(paramsFileName, argv[1]);
//     }

//     HGUStampRecog *recognizer = new HGUStampRecog(paramsFileName);

//     while(1){
//         std::cout << "Enter file name and dir to recognize : ";
//         std::cin >> filename;

//         if(strcmp(filename, "exit") == 0) break;

//         if(!checkFileDir(filename)){
//             std::cout << "There is no \"" << filename << "\"" << std::endl;
//             continue;
//         }
//         image = ReadBMP(filename, &img_width, &img_height);
//         result = recognizer->RecognizeStamp(img_width, img_height, image);

//         if(result == 0)
//             std::cout << "Fake!" << std::endl;
//         else
//             std::cout << "Real!" << std::endl;
//     }
//     return 0;
// }




/* This code for train  */
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
        
        if ((best_acc < (test_accuracy / LOOP_FOR_TEST))) {
            net->Save(filename);
        }
    }
    
    delete dataset;
    delete net;
    
    return 0;
}
