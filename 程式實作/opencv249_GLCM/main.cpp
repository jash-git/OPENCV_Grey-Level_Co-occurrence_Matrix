#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cstdio>
#include "GLCM.h"

#include <sys/timeb.h>
#if defined(WIN32)
    #define  TIMEB    _timeb
    #define  ftime    _ftime
    typedef __int64 TIME_T;
#else
    #define TIMEB timeb
    typedef long long TIME_T;
#endif

using namespace cv;
using namespace std;

//https://blog.csdn.net/lingtianyulong/article/details/53032034
void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}

int main()
{

	IplImage* img = cvLoadImage("Lena_original.jpg", 0);
	GLCM glcm;
	VecGLCM vec;
	GLCMFeatures features;
	glcm.initGLCM(vec);
	// 水平
	glcm.calGLCM(img, vec, GLCM::GLCM_HORIZATION);
	glcm.getGLCMFeatures(vec, features);
	// 垂直
	glcm.calGLCM(img, vec, GLCM::GLCM_VERTICAL);
	glcm.getGLCMFeatures(vec, features);
	// 45 度
	glcm.calGLCM(img, vec, GLCM::GLCM_ANGLE45);
	glcm.getGLCMFeatures(vec, features);
	// 135 度
	glcm.calGLCM(img, vec, GLCM::GLCM_ANGLE135);
	glcm.getGLCMFeatures(vec, features);

	cout << "asm = " << features.energy << endl;
	cout << "eng = " << features.entropy << endl;
	cout << "Con = " << features.contrast << endl;
    Pause();
    return 0;
}
