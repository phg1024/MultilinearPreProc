#include "common.h"
#include "BlendShape.h"

int main() {
	vector<BlendShape> shapes;

	const int nShapes = 150;

	const string path = "C:\\Users\\PhG\\Desktop\\Data\\FaceWarehouse_Data_0\\";
	const string foldername = "Tester_";
	const string bsfolder = "Blendshape";
	const string filename = "shape.bs";

	shapes.resize(nShapes);
	for(int i=1;i<=nShapes;i++) {
		stringstream ss;
		ss << path << foldername << i << "\\" << bsfolder + "\\" + filename;

		shapes[i].read(ss.str());
	}

	// create an order 3 tensor

}