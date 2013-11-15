#include "common.h"
#include "BlendShape.h"
#include "Tensor.hpp"

void testTensors() {
	Tensor1<float> t(10);
	for(int i=0;i<t.length();i++) {
		t(i) = (float)rand();
	}

	t.print();

	Tensor2<float> t2(5, 4);
	for(int i=0;i<t2.dim(0);i++) {
		for(int j=0;j<t2.dim(1);j++) {
			t2(i, j) = (float)rand();
		}
	}

	t2.print();

	t2.unfold().print();


	Tensor3<float> t3(2, 3, 4);
	for(int i=0;i<t3.dim(0);i++) {
		for(int j=0;j<t3.dim(1);j++) {
			for(int k=0;k<t3.dim(2);k++) {
				t3(i, j, k) = (float)(rand() % 32 );
			}			
		}
	}

	t3.print();

	cout << "unfold in mode 0:" << endl;
	Tensor2<float> t3_unfold0 = t3.unfold(0);
	t3_unfold0.print();
	Tensor3<float> t3new = Tensor3<float>::fold(t3_unfold0, 0, 2, 3, 4);
	t3new.print();

	cout << "unfold in mode 1:" << endl;
	Tensor2<float> t3_unfold1 = t3.unfold(1);
	t3_unfold1.print();
	Tensor3<float> t3new2 = Tensor3<float>::fold(t3_unfold1, 1, 2, 3, 4);
	t3new2.print();
}

int main() {	
	
	testTensors();
	system("pause");
	return 0;
	

	vector<BlendShape> shapes;

	const int nShapes = 150;			// 150 identity
	const int nExprs = 47;				// 46 expressions + 1 neutral
	const int nVerts = 11510;			// 11510 vertices for each mesh

	const string path = "C:\\Users\\PhG\\Desktop\\Data\\FaceWarehouse_Data_0\\";
	const string foldername = "Tester_";
	const string bsfolder = "Blendshape";
	const string filename = "shape.bs";

	shapes.resize(nShapes);
	for(int i=0;i<nShapes;i++) {
		stringstream ss;
		ss << path << foldername << (i+1) << "\\" << bsfolder + "\\" + filename;

		shapes[i].read(ss.str());
	}

	// create an order 3 tensor for the blend shapes
	int nCoords = nVerts * 3;
	Tensor3<float> t(nShapes, nExprs, nCoords);

	// fill in the data
	for(int i=0;i<shapes.size();i++) {
		const BlendShape& bsi = shapes[i];
		for(int j=0;j<bsi.expressionCount();j++) {
			const BlendShape::shape_t& bsij = bsi.expression(j);
			
			for(int k=0, cidx = 0;k<nVerts;k++, cidx+=3) {
				const BlendShape::vert_t& v = bsij[k];

				t(i, j, cidx) = v.x;
				t(i, j, cidx+1) = v.y;
				t(i, j, cidx+2) = v.z;
			}
		}
	}

	cout << "Tensor assembled." << endl;

	t.write("blendshape.tensor");

	system("pause");

	return 0;
}