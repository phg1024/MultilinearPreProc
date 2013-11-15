#pragma once

template <typename T>
class Tensor1
{
public:
	Tensor1(void){}
	Tensor1(int n):n(n) { data.resize(n); }
	~Tensor1(){}

	const T& operator()(int i) const{
		return data[i];
	}
	T& operator()(int i) {
		return data[i];
	}

	int length() const {
		return n;
	}
	void resize(int size){
		n = size;
		data.resize(n);
	}

	void print() const{
		cout << (*this);
	}
private:
	int n;
	vector<T> data;
};

template <typename T>
ostream& operator<<(ostream& os, const Tensor1<T>& t) {
	for(int i=0;i<t.length();i++) {
		os << t(i) << ((i==t.length()-1)?'\n':' ');
	}
	return os;
}

// Order 2 tensor, basically a matrix
template <typename T>
class Tensor2
{
public:
	Tensor2(void){}
	Tensor2(int m, int n){
		d[0] = m; d[1] = n;
		data.resize(m);
		for_each(data.begin(), data.end(), [=](Tensor1<T>& t){
			t.resize(n);
		});
	}
	~Tensor2(){}

	const Tensor1<T>& operator()(int i) const {
		return data[i];
	}

	Tensor1<T>& operator()(int i) {
		return data[i];
	}

	const T& operator()(int i, int j) const {
		return data[i](j);
	}
	T& operator()(int i, int j) {
		return data[i](j);
	}

	int dim(int mid) const{
		return d[mid];
	}

	int rows() { return d[0]; }
	int cols() { return d[1]; }

	void resize(int r, int c){
		d[0] = r;
		d[1] = c;
		data.resize(r);
		for_each(data.begin(), data.end(), [=](Tensor1<T>& t){
			t.resize(c);
		});
	}

	Tensor1<T> unfold() const{
		Tensor1<T> t(d[0] * d[1]);

		for(int i=0, idx=0;i<d[0];i++) {
			for(int j=0;j<d[1];j++, idx++) {
				t(idx) = data[i](j);
			}
		}

		return t;
	}

	void print() const{
		for(int i=0;i<d[0];i++) {
			data[i].print();
		}
	}

	template <typename TT>
	friend ostream& operator<<(ostream& os, const Tensor2<TT>& t);

private:
	int d[2];
	vector<Tensor1<T>> data;
};

template <typename T>
ostream& operator<<(ostream& os, const Tensor2<T>& t) {
	for(int i=0;i<t.dim(0);i++) {
		os << t.data[i] << endl;
	}
	return os;
}

// Order 3 tensor
template <typename T>
class Tensor3
{
public:
	Tensor3(void){}
	Tensor3(int l, int m, int n){
		d[0] = l; d[1] = m; d[2] = n;
		data.resize(l);
		for_each(data.begin(), data.end(), [=](Tensor2<T>& t){
			t.resize(m, n);
		});
	}
	~Tensor3(void){}

	const Tensor2<T>& operator()(int i) const {
		return data[i];
	}
	Tensor2<T>& operator()(int i) {
		return data[i];
	}

	const Tensor1<T>& operator()(int i, int j) const {
		return data[i](j);
	}
	Tensor1<T>& operator()(int i, int j) {
		return data[i](j);
	}

	const T& operator()(int i, int j, int k) const{
		return data[i](j, k);
	}
	T& operator()(int i, int j, int k) {
		return data[i](j, k);
	}

	int dim(int mid) const{
		return d[mid];
	}

	Tensor2<T> unfold(int mid) const {
		switch( mid ) {
		case 0:{
					Tensor2<T> t(d[0], d[1] * d[2]);
					// fill in the values
					for(int i=0;i<d[0];i++) {
						const Tensor2<T>& ti = data[i];
						Tensor1<T> ti_unfold = ti.unfold();

						// copy every thing into t
						for(int j=0;j<d[1] * d[2];j++) {
							t(i, j) = ti_unfold(j);
						}
					}
					return t;
			   }		
		case 1:{			
					Tensor2<T> t(d[1], d[0] * d[2]);
					// fill in the values
					for(int i=0;i<d[0];i++) {
						// copy every thing into t
						for(int j=0;j<d[1];j++) {
							for(int k=0, offset=0;k<d[2];k++, offset+=d[0]) {
								t(j, offset+i) = (*this)(i, j, k);
							}
						}
					}
					return t;
			   }
		case 2: {
					Tensor2<T> t(d[2], d[0] * d[1]);
					// fill in the values
					for(int i=0, offset=0;i<d[0];i++, offset+=d[1]) {
						const Tensor2<T>& ti = data[i];
						for(int j=0;j<d[1];j++) {
							for(int k=0;k<d[2];k++) {
								t(k, j+offset) = ti(j, k);
							}
						}
					}
					return t;
				}		
		default: {
					throw "Invalid mode!";
				 }
		}
	}

	// mode product with a vector
	Tensor2<T> modeProduct(const Tensor1<T>& v, int mid) {
		switch( mid ) {
		case 0:
			{
				break;
			}
		case 1:
			{
				break;
			}
		case 2:
			{
				break;
			}
		default:
			throw "Invalid mode!";
		}
	}

	// mode product with a matrix
	Tensor3<T> modeProduct(const Tensor2<T>& M, int mid) {
		switch( mid ) {
		case 0:
			{
				break;
			}
		case 1:
			{
				break;
			}
		case 2:
			{
				break;
			}
		}
	}

	// ignore the third mode
	tuple<Tensor3<T>, vector<Tensor2<T>>> svd() const {
		// unfold the tensor in mode 0 and compute the svd of the unfolded matrix

		// unfold the tensor in mode 1 and compute the svd of the unfolded matrix

		// core tensor is then defined as T x0 U(0)' x0 U(1)'

		// return the core, U(0) and U(1)
	}

	void print() {
		for(int i=0;i<d[0];i++) {
			data[i].print();
		}
	}

	bool write(const string& filename) {
		try {
			cout << "writing tensor to file " << filename << endl;
			fstream fout;
			fout.open(filename, ios::out);

			fout.write(reinterpret_cast<char*>(&(d[0])), sizeof(int)*3);

			for(int i=0;i<d[0];i++) {
				const Tensor2<T>& ti = data[i];
				for(int j=0;j<d[1];j++) {
					fout.write(reinterpret_cast<const char*>(&(ti(j)(0))), sizeof(T)*d[2]);
				}				
			}

			fout.close();

			cout << "done." << endl;
			return true;
		}
		catch(...) {
			cerr << "Failed to write tensor to file " << filename << endl;
			return false;
		}
	}

private:
	// mode 0, mode 1, mode 2
	int d[3];
	vector<Tensor2<T>> data;
};

