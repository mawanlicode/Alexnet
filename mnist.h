#ifndef __Mnist_
#define __Mnist_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

//单组数据及保存数据的链表
typedef struct Signaldata{
	int c;           // 数据长
	int r;           // 数据款默认为2
	float** IQData; // IQ数据动态数组
}Signaldata;

typedef struct SignaldataArr{
	int dataNum;        // 数据的数量
	Signaldata* DataPtr;  // 数据指针
}SignaldataArrS, *DataArr; //数据存储

typedef struct DataLabel{
	int l;            // 输出标签长度
	float* LabelData; // 输出标签数据
}MnistLabel;

typedef struct DataLabelArr{
	int LabelNum;
	MnistLabel* LabelPtr;
} MnistLabelArrS, *LabelArr; //存储数据标签的数组

LabelArr read_Lable(const char* filename); 

DataArr read_Data(const char* filename);

LabelArr read_Lable_ar();

DataArr read_Data_ar();

void save_Img(DataArr imgarr, char* filedir);

#endif