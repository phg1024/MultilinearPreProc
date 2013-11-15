#pragma once

template <typename T>
class Tensor1
{
public:
	Tensor1(void){}
	Tensor1(int n):n(n) { data.resize(n); }
	~Tensor1(){}

	const T& operator()(int i) {
		return data[i];
	}
	T operator()(int i) {
		return data[i];
	}

	int length() {
		return n;
	}
	void resize(int size){
		n = size;
		data.resize(n);
	}
private:
	int n;
	valarray<T> data;
};

// Order 2 tensor, basically a matrix
template <typename T>
class Tensor2
{
public:
	Tensor2(void);
	Tensor2(int m, int n):m(m), n(n) {
		data.resize(m);
		for_each(data.begin(), data.end(), [](Tensor1<T>& t){
			t.resize(n);
		});
	}
	~Tensor2();

	const T& operator()(int i, int j) {
		return data[i](j);
	}
	T operator()(int i, int j) {
		return data[i](j);
	}

	int rows() { return m; }
	int cols() { return n; }

	void resize(int r, int c){
		m = r;
		n = c;
		data.resize(m);
		for_each(data.begin(), data.end(), [](Tensor1<T>& t){
			t.resize(n);
		});
	}

	Tensor1 unfold() {
		Tensor1 t(m * n);

		for(int i=0, idx=0;i<m;i++) {
			for(int j=0;j<n;j++) {
				t(idx) = data[m](n);
			}
		}

		return t;
	}

private:
	int m, n;
	valarray<Tensor1<T>> data;
};

// Order 3 tensor
template <typename T>
class Tensor3
{
public:
	Tensor3(void);
	Tensor3(int l, int m, int n):l(l), m(m), n(n) {
		data.resize(l);
		for_each(data.begin(), data.end(), [](Tensor2<T>& t){
			t.resize(m, n);
		});
	}
	~Tensor3(void);

	const T& operator()(int i, int j, int k) {
		return data[i](j, k);
	}
	T operator()(int i, int j, int k) {
		return data[i](j, k);
	}

	Tensor2 unfold(int mid) const {
		if( mid == 1 ) {
			Tensor2 t(l, m * n);
			// fill in the values
			for(int i=0;i<l;i++) {
				const Tensor2<T>& ti = data[i];
				Tensor1<T> ti_unfold = ti.unfold();
				
				// copy every thing into t
				for(int j=0;j<m * n;j++) {
					t(i, j) = ti_unfold(j);
				}
			}
			return t;
		}
		else if( mid == 0 ) {
			Tensor2 t(m, l * n);
			// fill in the values
			for(int i=0, offset=0;i<l;i++, offset+=l*n) {
				const Tensor2<T>& ti = data[i];
				for(int j=0;j<m;j++) {
					for(int k=0;k<n;k++) {
						t(j, k+offset) = ti(j, k);
					}
				}
			}
		}
		else if( mid == 2 ) {
			// not supported yet
			throw lazy_exception("Not supported yet.");
		}
	}

private:
	// mode 0, mode 1, mode 2
	int l, m, n;
	valarray<Tensor2<T>> data;
};

