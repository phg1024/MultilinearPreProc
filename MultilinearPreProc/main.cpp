#include "common.h"
#include "utility.hpp"
#include "BlendShape.h"
#include "Tensor.hpp"

void testTensors() {
	Tensor1<float> t(10);
	for(int i=0;i<t.length();i++) {
		t(i) = (float)rand();
	}

	t.print();

	Tensor2<float> t2(3, 2);
	for(int i=0;i<t2.dim(0);i++) {
		for(int j=0;j<t2.dim(1);j++) {
			t2(i, j) = (float)(rand() % 16);
		}
	}

	t2.print("T2");

	t2.unfold().print();


	Tensor3<float> t3(2, 3, 4);
	for(int i=0;i<t3.dim(0);i++) {
		for(int j=0;j<t3.dim(1);j++) {
			for(int k=0;k<t3.dim(2);k++) {
				t3(i, j, k) = (float)(rand() % 32 );
			}			
		}
	}

	t3.print("T3");

	cout << "unfold in mode 0:" << endl;
	Tensor2<float> t3_unfold0 = t3.unfold(0);
	t3_unfold0.print("T30");
	Tensor3<float> t3new = Tensor3<float>::fold(t3_unfold0, 0, 2, 3, 4);
	t3new.print("fold back");

	cout << "unfold in mode 1:" << endl;
	Tensor2<float> t3_unfold1 = t3.unfold(1);
	t3_unfold1.print("T31");
	Tensor3<float> t3new2 = Tensor3<float>::fold(t3_unfold1, 1, 2, 3, 4);
	t3new2.print("fold back");

	cout << "unfold in mode 1:" << endl;
	Tensor2<float> t3_unfold2 = t3.unfold(2);
	t3_unfold2.print("T32");
	Tensor3<float> t3new3 = Tensor3<float>::fold(t3_unfold2, 2, 2, 3, 4);
	t3new3.print("fold back");

	cout << "mode product" << endl;
	Tensor3<float> tm0 = t3.modeProduct(t2, 0);
	tm0.print("TM0");	

	Tensor2<float> t22(3, 3);
	for(int i=0;i<t22.dim(0);i++) {
		for(int j=0;j<t22.dim(1);j++) {
			t22(i, j) = (float)(rand() % 16);
		}
	}
	t22.print("T22");

	Tensor3<float> tm1 = t3.modeProduct(t22, 1);
	tm1.print("TM1");

	Tensor2<float> t23(3, 4);
	for(int i=0;i<t23.dim(0);i++) {
		for(int j=0;j<t23.dim(1);j++) {
			t23(i, j) = (float)(rand() % 16);
		}
	}
	t23.print("T23");

	Tensor3<float> tm2 = t3.modeProduct(t23, 2);
	tm2.print("TM2");

	auto comp = t3.svd();
	auto tcore = std::get<0>(comp);
	auto tu0 = std::get<1>(comp);
	auto tu1 = std::get<2>(comp);
	auto tu2 = std::get<3>(comp);

	tcore.print("core");
	tu0.print("u0");
	tu1.print("u1");
	tu2.print("u2");

	auto trecon = tcore.modeProduct(tu0, 0)
		.modeProduct(tu1, 1)
		.modeProduct(tu2, 2);
	trecon.print("recon");
	t3.print("ref");

	int ms[3] = {0, 1, 2};
	int ds[3] = {2, 3, 4};
	vector<int> modes(ms, ms+3);
	vector<int> dims(ds, ds+3);
	auto comp2 = t3.svd(modes, dims);

	tcore = std::get<0>(comp2);
	auto tus = std::get<1>(comp2);
	tcore.print("core");
	trecon = tcore;
	for(int i=0;i<modes.size();i++) {
		auto tui = tus[i];
		trecon = trecon.modeProduct(tui, modes[i]);
		tui.print("ui");
	}

	trecon.print("recon");
	t3.print("ref");
}

int main() {	
	
	
	//testTensors();
	//system("pause");
	//return 0;
	

	vector<BlendShape> shapes;

	const int nShapes = 150;			// 150 identity
	const int nExprs = 47;				// 46 expressions + 1 neutral
	const int nVerts = 11510;			// 11510 vertices for each mesh

	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\FaceWarehouse_Data_0\\";
	const string foldername = "Tester_";
	const string bsfolder = "Blendshape";
	const string filename = "shape.bs";

	shapes.resize(nShapes);
	for(int i=0;i<nShapes;i++) {
		stringstream ss;
		ss << path << foldername << (i+1) << "\\" << bsfolder + "\\" + filename;

		shapes[i].read(ss.str());
	}
	int nCoords = nVerts * 3;

	// create an order 3 tensor for the blend shapes	
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

	// perform svd to get core tensor
	cout << "Performing SVD on the blendshapes ..." << endl;
	int ms[2] = {0, 1};		// only the first two modes
	int ds[2] = {50, 25};	// pick 50 for identity and 25 for expression
	vector<int> modes(ms, ms+2);
	vector<int> dims(ds, ds+2);
	auto comp2 = t.svd(modes, dims);
	cout << "SVD done." << endl;	

	auto tcore = std::get<0>(comp2);
	auto tus = std::get<1>(comp2);	
	cout << "writing core tensor ..." << endl;
	tcore.write("blendshape_core.tensor");
	cout << "writing U tensors ..." << endl;
	for(int i=0;i<tus.size();i++) {
		tus[i].write("blendshape_u_" + toString(ms[i]) + ".tensor");
	}
	
	cout << "Validation begins ..." << endl;
	Tensor3<float> tin;
	tin.read("blendshape_core.tensor");

	cout << "Core tensor dimensions = " 
		<< tin.dim(0) << "x"
		<< tin.dim(1) << "x"
		<< tin.dim(2) << endl;

	float maxDiffio = 0;

	for(int i=0;i<tin.dim(0);i++) {
		for(int j=0;j<tin.dim(1);j++) {
			for(int k=0;k<tin.dim(2);k++) {
				maxDiffio = std::max(fabs(tin(i, j, k) - tcore(i, j, k)), maxDiffio);
			}
		}
	}
	cout << "Max difference io = " << maxDiffio << endl;

	tin = tin.modeProduct(tus[0], 0).modeProduct(tus[1], 1);

	cout << "Dimensions = " 
		<< tin.dim(0) << "x"
		<< tin.dim(1) << "x"
		<< tin.dim(2) << endl;
	float maxDiff = 0;

	for(int i=0;i<tin.dim(0);i++) {
		for(int j=0;j<tin.dim(1);j++) {
			for(int k=0;k<tin.dim(2);k++) {
				maxDiff = std::max(fabs(tin(i, j, k) - t(i, j, k)), maxDiff);
			}
		}
	}

	cout << "Max difference = " << maxDiff << endl;
	cout << "done" << endl;

	system("pause");
	return 0;
}