#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <queue>
#include <ctime>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "../../WICWIU_src/Tensor.hpp"

using namespace std;

#define O_listFile "../../../OX/O/O_list.txt"
#define X_listFile "../../../OX/X/X_list.txt"
#define NUMBER_OF_O 1343
#define NUMBER_OF_X 882
#define NUMBER_OF_TRAIN_IMAGE 1558
#define NUMBER_OF_TEST_IMAGE 667
#define NUMBER_OF_CHANNEL 1
#define WIDTH 30
#define HEIGHT 30
#define NUMBER_OF_CLASS 2

#define BYTE unsigned char
#define DWORD int
#define WORD unsigned short
#define LONG int

#pragma pack(push,1)

typedef struct tagBITMAPFILEHEADER {
    short bfType;       
    int   bfSize;       
    short bfReserved1;
    short bfReserved2;
    int   bfOffBits;    
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER{
    DWORD biSize;          //현 구조체의 크기
    LONG biWidth;          //이미지의 가로 크기
    LONG biHeight;         //이미지의 세로 크기
    WORD biPlanes;         //플레인수
    WORD biBitCount;       //비트 수
    DWORD biCompression;   //압축 유무
    DWORD biSizeImage;     //이미지 크기
    LONG biXPelsPerMeter;  //미터당 가로 픽셀
    LONG biYPelsPerMeter;  //미터당 세로 픽셀
    DWORD biClrUsed;       //컬러 사용 유무
    DWORD biClrImportant;  //중요하게 사용하는 색
}BITMAPINFOHEADER;

typedef struct tagRGBQUAD {
    BYTE   rgbBlue;
    BYTE   rgbGreen;
    BYTE   rgbRed;
    BYTE   rgbReserved;
}  RGBQUAD;

#pragma pack(pop)

template <typename DTYPE>
class BMPDataSet{
private:
    int m_batchSize;

    vector<char *> m_imageNameList;
    vector<int> m_imageNumber;
    vector<int> m_TrainImageIdx;
    vector<int> m_TestImageIdx;

    // batch Tensor << before concatenate
    queue<Tensor<DTYPE> *> *m_aaSetOfImage; // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabel; // size : batch size

    queue<Tensor<DTYPE> *> *m_aaSetOfImageForConcatenate; // size : batch size
    queue<Tensor<DTYPE> *> *m_aaSetOfLabelForConcatenate; // size : batch size

    // Storage for preprocessed Tensor
    queue<Tensor<DTYPE> **> *m_aaQForTrainData; 
    queue<Tensor<DTYPE> **> *m_aaQForTestData;

public:
    BMPDataSet(int batchSize){
        m_batchSize = batchSize;

        Alloc();
        CreateDataNumber();
        GetDataList();
    }

    virtual ~BMPDataSet(){
        Delete();
    }

    void Alloc(){
        m_aaSetOfImage = new queue<Tensor<DTYPE> *>(); 
        m_aaSetOfLabel = new queue<Tensor<DTYPE> *>();

        m_aaSetOfImageForConcatenate = new queue<Tensor<DTYPE> *>(); 
        m_aaSetOfLabelForConcatenate = new queue<Tensor<DTYPE> *>();

        m_aaQForTrainData = new queue<Tensor<DTYPE> **>();
        m_aaQForTestData = new queue<Tensor<DTYPE> **>();
    }

    void Delete() {
        if (m_aaSetOfImage){
            if (m_aaSetOfImage->size() != 0){
                int numOfTensor = m_aaSetOfImage->size();

                for (int i = 0; i < numOfTensor; i++){
                    delete m_aaSetOfImage->front();
                    m_aaSetOfImage->front() = NULL;
                    m_aaSetOfImage->pop();
                }
            }
            delete m_aaSetOfImage;
            m_aaSetOfImage = NULL;
        }

        if (m_aaSetOfLabel){
            if (m_aaSetOfLabel->size() != 0){
                int numOfTensor = m_aaSetOfLabel->size();

                for (int i = 0; i < numOfTensor; i++){
                    delete m_aaSetOfLabel->front();
                    m_aaSetOfLabel->front() = NULL;
                    m_aaSetOfLabel->pop();
                }
            }
            delete m_aaSetOfLabel;
            m_aaSetOfLabel = NULL;
        }

        if (m_aaSetOfImageForConcatenate){
            if (m_aaSetOfImageForConcatenate->size() != 0){
                int numOfTensor = m_aaSetOfImageForConcatenate->size();

                for (int i = 0; i < numOfTensor; i++){
                    delete m_aaSetOfImageForConcatenate->front();
                    m_aaSetOfImageForConcatenate->front() = NULL;
                    m_aaSetOfImageForConcatenate->pop();
                }
            }
            delete m_aaSetOfImageForConcatenate;
            m_aaSetOfImageForConcatenate = NULL;
        }

        if (m_aaSetOfLabelForConcatenate){
            if (m_aaSetOfLabelForConcatenate->size() != 0){
                int numOfTensor = m_aaSetOfLabelForConcatenate->size();

                for (int i = 0; i < numOfTensor; i++){
                    delete m_aaSetOfLabelForConcatenate->front();
                    m_aaSetOfLabelForConcatenate->front() = NULL;
                    m_aaSetOfLabelForConcatenate->pop();
                }
            }
            delete m_aaSetOfLabelForConcatenate;
            m_aaSetOfLabelForConcatenate = NULL;
        }

        if (m_aaQForTrainData){
            if (m_aaQForTrainData->size() != 0){
                int numOfTensor = m_aaQForTrainData->size();

                for (int i = 0; i < numOfTensor; i++){
                    Tensor<DTYPE> **temp = m_aaQForTrainData->front();
                    m_aaQForTrainData->pop();
                    delete temp[0];
                    delete temp[1];
                    delete[] temp;
                    temp = NULL;
                }
            }
            delete m_aaQForTrainData;
            m_aaQForTrainData = NULL;
        }

        if (m_aaQForTestData){
            if (m_aaQForTestData->size() != 0){
                int numOfTensor = m_aaQForTestData->size();

                for (int i = 0; i < numOfTensor; i++){
                    Tensor<DTYPE> **temp = m_aaQForTestData->front();
                    m_aaQForTestData->pop();
                    delete temp[0];
                    delete temp[1];
                    delete[] temp;
                    temp = NULL;
                }
            }
            delete m_aaQForTestData;
            m_aaQForTestData = NULL;
        }
    }

    void CreateDataNumber(){
        for (int i = 0; i < NUMBER_OF_O + NUMBER_OF_X; i++)
            m_imageNumber.push_back(i);

        std::random_shuffle(m_imageNumber.begin(), m_imageNumber.end());

        for (int i = 0; i < NUMBER_OF_TRAIN_IMAGE; i++)
            m_TrainImageIdx.push_back(m_imageNumber[i]);

        for (int i = 0; i < NUMBER_OF_TEST_IMAGE; i++)
            m_TestImageIdx.push_back(m_imageNumber[i + NUMBER_OF_TRAIN_IMAGE]);
    }

    void GetDataList(){
        FILE *O_fp = fopen(O_listFile, "r");
        char strTemp[255];
        char *pStr;

        if (O_fp != NULL){
            while (!feof(O_fp)){
                pStr = fgets(strTemp, sizeof(strTemp), O_fp);
                char *temp = (char *)malloc(sizeof(char) * 50);
                if (pStr){
                    strcpy(temp, pStr);
                    for (int i = 0; i < strlen(temp); i++)
                        if (temp[i] == '\n')
                            temp[i] = 0;

                    m_imageNameList.push_back(temp);
                }
            }
        }
        else{
            printf("file list open fail\n");
        }

        FILE *X_fp = fopen(X_listFile, "r");
        if (X_fp != NULL){
            while (!feof(X_fp)){
                pStr = fgets(strTemp, sizeof(strTemp), X_fp);
                char *temp = (char *)malloc(sizeof(char) * 50);
                if (pStr){
                    strcpy(temp, pStr);
                    for (int i = 0; i < strlen(temp); i++)
                        if (temp[i] == '\n')
                            temp[i] = 0;

                    m_imageNameList.push_back(temp);
                }
            }
        }
        else{
            printf("file list open fail\n");
        }

        fclose(O_fp);
        fclose(X_fp);
    }

    //image 1개를 읽어드리는
    Tensor<DTYPE> *Image2Tensor(char *imageDir){
        FILE *input;
        unsigned char *bmp;
        long size;
        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, WIDTH, HEIGHT);

        input = fopen(imageDir, "rb");
        if (input == NULL){
            printf("Can not open the file\n");
            cout << imageDir << endl;
        }
        fseek(input, 0, SEEK_END); 
        size = ftell(input);       
        rewind(input);             

        bmp = new unsigned char[size]; 
        int dum = fread(bmp, 1, size, input);    
        fclose(input);

        
        BITMAPFILEHEADER *bf = (BITMAPFILEHEADER *)bmp;
        BITMAPINFOHEADER *bi = (BITMAPINFOHEADER *)(bmp + sizeof(*bf));
        RGBQUAD *gr = (RGBQUAD *)(bmp + sizeof(*bf) + sizeof(*bi)); 
        unsigned char *img = bmp + bf->bfOffBits;                   

        
        if (bf->bfType != 0x4D42){
            printf("Not bitmap file!");
        }
        if (bi->biBitCount != 8){
            printf("Bad file format!");
        }

        
        int w = bi->biWidth;
        if (w % 4)
            w += 4 - bi->biWidth % 4;

        
        float bmptxt[WIDTH][HEIGHT]; 
        unsigned char B, G, R;
        int p;

        // 배열에 저장
        for (int i = 0; i < bi->biHeight; i++){
            for (int j = 0; j < bi->biWidth; j++){
                p = img[(bi->biHeight - i - 1) * w + j];
                B = gr[p].rgbBlue;
                G = gr[p].rgbGreen;
                R = gr[p].rgbRed;
                bmptxt[i][j] = (((114 * R + 587 * G + 299 * B) / 1000) / 255.0) ;
                (*temp)[Index5D(temp->GetShape(), 0, 0, 0, j, i)] = bmptxt[i][j];
                // printf("%.3f ", bmptxt[i][j]);
            }
            // printf("\n");
        }

        free(bmp);
        return temp;
    }

    Tensor<DTYPE> *Label2Tensor(int classNum){
        Tensor<DTYPE> *temp = Tensor<DTYPE>::Zeros(1, 1, 1, 1, NUMBER_OF_CLASS);
        (*temp)[classNum] = (DTYPE)1;
        return temp;
    }

    Tensor<DTYPE> *ConcatenateImage(queue<Tensor<DTYPE> *> *setOfImage){
        int singleImageSize = setOfImage->front()->GetCapacity();

        Tensor<DTYPE> *result = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, singleImageSize);
        Tensor<DTYPE> *singleImage = NULL;

        for (int batchNum = 0; batchNum < m_batchSize; batchNum++)
        {
            singleImage = setOfImage->front();
            setOfImage->pop();
            for (int idxOfImage = 0; idxOfImage < singleImageSize; idxOfImage++)
            {
                int idxOfResult = batchNum * singleImageSize + idxOfImage;
                (*result)[idxOfResult] = (*singleImage)[idxOfImage];
            }

            delete singleImage;
            singleImage = NULL;
        }
        return result;
    }

    Tensor<DTYPE> *ConcatenateLabel(queue<Tensor<DTYPE> *> *setOfLabel){
        Tensor<DTYPE> *result = Tensor<DTYPE>::Zeros(1, m_batchSize, 1, 1, NUMBER_OF_CLASS);
        Tensor<DTYPE> *singleLabel = NULL;

        for (int batchNum = 0; batchNum < m_batchSize; batchNum++){
            singleLabel = setOfLabel->front();
            setOfLabel->pop();

            for (int idxOfLabel = 0; idxOfLabel < NUMBER_OF_CLASS; idxOfLabel++){
                int idxOfResult = batchNum * NUMBER_OF_CLASS + idxOfLabel;
                (*result)[idxOfResult] = (*singleLabel)[idxOfLabel];
            }

            delete singleLabel;
            singleLabel = NULL;
        }
        return result;
    }
    int AddData2TrainBuffer(Tensor<DTYPE> *setOfImage, Tensor<DTYPE> *setOfLabel){
        Tensor<DTYPE> **result = new Tensor<DTYPE> *[2];

        result[0] = setOfImage;
        result[1] = setOfLabel;

        m_aaQForTrainData->push(result);

        return TRUE;
    }

    int AddData2TestBuffer(Tensor<DTYPE> *setOfImage, Tensor<DTYPE> *setOfLabel){
        Tensor<DTYPE> **result = new Tensor<DTYPE> *[2];

        result[0] = setOfImage;
        result[1] = setOfLabel;

        m_aaQForTestData->push(result);

        return TRUE;
    }

    //데이터와 label을 vector에 넣는다.
    void LoadTrainImage(){
        int numOfbatchBlock = NUMBER_OF_TRAIN_IMAGE / m_batchSize;
        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;
        Tensor<DTYPE> *temp = NULL;

        std::random_shuffle(m_TrainImageIdx.begin(), m_TrainImageIdx.end());

        for (int i = 0; i < numOfbatchBlock; i++){
            for (int j = 0; j < m_batchSize; j++){
                if (m_TrainImageIdx[i * m_batchSize + j] < NUMBER_OF_O){
                    char O_Dir[50] = "../../../OX/O/";
                    strcat(O_Dir, m_imageNameList[m_TrainImageIdx[i * m_batchSize + j]]);
                    temp = this->Image2Tensor(O_Dir);
                    m_aaSetOfImage->push(temp);
                    m_aaSetOfLabel->push(this->Label2Tensor(1));
                }
                else{
                    char X_Dir[50] = "../../../OX/X/";
                    strcat(X_Dir, m_imageNameList[m_TrainImageIdx[i * m_batchSize + j]]);
                    temp = this->Image2Tensor(X_Dir);
                    m_aaSetOfImage->push(this->Image2Tensor(X_Dir));
                    m_aaSetOfLabel->push(this->Label2Tensor(0));
                }
            }
            preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
            preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);

            this->AddData2TrainBuffer(preprocessedImages, preprocessedLabels);
        }
    }

    void LoadTestImage(){
        int numOfbatchBlock = NUMBER_OF_TEST_IMAGE / m_batchSize;
        Tensor<DTYPE> *preprocessedImages = NULL;
        Tensor<DTYPE> *preprocessedLabels = NULL;
        Tensor<DTYPE> *temp = NULL;

        std::random_shuffle(m_TestImageIdx.begin(), m_TestImageIdx.end());

        for (int i = 0; i < numOfbatchBlock; i++){
            for (int j = 0; j < m_batchSize; j++){
                if (m_TestImageIdx[i * m_batchSize + j] < NUMBER_OF_O){
                    char O_Dir[50] = "../../../OX/O/";
                    strcat(O_Dir, m_imageNameList[m_TestImageIdx[i * m_batchSize + j]]);
                    temp = this->Image2Tensor(O_Dir);
                    m_aaSetOfImage->push(temp);
                    m_aaSetOfLabel->push(this->Label2Tensor(1));
                }
                else{
                    char X_Dir[50] = "../../../OX/X/";
                    strcat(X_Dir, m_imageNameList[m_TestImageIdx[i * m_batchSize + j]]);
                    temp = this->Image2Tensor(X_Dir);
                    m_aaSetOfImage->push(this->Image2Tensor(X_Dir));
                    m_aaSetOfLabel->push(this->Label2Tensor(0));
                }
            }
            preprocessedImages = this->ConcatenateImage(m_aaSetOfImage);
            preprocessedLabels = this->ConcatenateLabel(m_aaSetOfLabel);

            this->AddData2TestBuffer(preprocessedImages, preprocessedLabels);
        }
    }

    void prepareData(){
        this->LoadTrainImage();
        this->LoadTestImage();
    }

    Tensor<DTYPE> **GetTrainDataFromBuffer(){
        Tensor<DTYPE> **result = m_aaQForTrainData->front();
        m_aaQForTrainData->pop();
        return result;
    }
    Tensor<DTYPE> **GetTestDataFromBuffer(){
        Tensor<DTYPE> **result = m_aaQForTestData->front();
        m_aaQForTestData->pop();
        return result;
    }
};

template <typename DTYPE>
BMPDataSet<DTYPE> *CreatBMPDataSet(int batchSize)
{
    BMPDataSet<DTYPE> *dataset = new BMPDataSet<DTYPE>(batchSize);
    return dataset;
}
