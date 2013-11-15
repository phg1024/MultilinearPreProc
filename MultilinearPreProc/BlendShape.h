#pragma once

#include "common.h"

class BlendShape
{
public:
	struct vert_t {
		float x, y, z;
	};
	typedef vector<vert_t> shape_t;

	BlendShape(void);
	~BlendShape(void);

public:
	bool read(const string& filename);

protected:
	void drawShape(const shape_t& s);

private:
	int nVerts;
	int nShapes;
	int nFaces;

	shape_t neutralExpr;
	vector<shape_t> exprList;
};

