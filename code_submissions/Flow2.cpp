//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kussa Richárd
// Neptun : RONAOF
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
using namespace std;

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

class Camera {
	vec2 wCenter;
	vec2 wSize;
public:
	Camera(vec2 wc, vec2 ws) { 
		wCenter = wc;
		wSize = ws;
	};
	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }
	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }
	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera camera(vec2(0, 0), vec2(30, 30));
const int nTessVertices = 100;

class Curve {
	unsigned int vaoCurve = 0, vboCurve = 0, vaoCPS = 0, vboCPS = 0;
protected:
	vector<vec2> cps;
public:
	Curve() {
		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);
		glGenBuffers(1, &vboCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);

		glGenVertexArrays(1, &vaoCPS);
		glBindVertexArray(vaoCPS);
		glGenBuffers(1, &vboCPS);
		glBindBuffer(GL_ARRAY_BUFFER, vboCPS);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
	}
	~Curve() {
		glDeleteBuffers(1, &vboCPS);
		glDeleteBuffers(1, &vaoCPS);
		glDeleteBuffers(1, &vboCurve);
		glDeleteBuffers(1, &vaoCurve);
	}	
	virtual vec2 r(float t) = 0;
	virtual float tStart() = 0;
	virtual float tEnd() = 0;
	virtual void AddControlPoint(vec2 p) {
		vec4 temp = vec4(p.x, p.y, 0, 1) * camera.Pinv() * camera.Vinv();
		cps.push_back(vec2(temp.x, temp.y));
	}
	int PickControlPoint(vec2 p) {
		vec4 temp = vec4(p.x, p.y, 0, 1) * camera.Pinv() * camera.Vinv();
		vec2 cursorPoint = vec2(temp.x, temp.y);
		for (int i = 0; i < cps.size(); i++) {
			if (dot(cps[i] - cursorPoint, cps[i] - cursorPoint) < 0.1) return i;
		}
		return -1;
	}
	void MoveControlPoint(int i, vec2 p) {
		vec4 temp = vec4(p.x, p.y, 0, 1) * camera.Pinv() * camera.Vinv();
		cps[i] = vec2(temp.x, temp.y);
	}

	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();
		gpuProgram.setUniform(VPTransform, "MVP");
		if (cps.size() > 0) {
			glBindVertexArray(vaoCPS);
			glBindBuffer(GL_ARRAY_BUFFER, vboCPS);
			glBufferData(GL_ARRAY_BUFFER, cps.size() * sizeof(vec2), &cps[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 0, 0), "color");
			glDrawArrays(GL_POINTS, 0, cps.size());
		}
		if (cps.size() > 1) {
			vector<vec2> vtxData;
			for (int i = 0; i < nTessVertices; i++) {
				float tNormalized = (float)i / (nTessVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec2 vtx = r(t);
				vtxData.push_back(vtx);
			}
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(vec2), &vtxData[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 1, 0), "color");
			glDrawArrays(GL_LINE_STRIP, 0, nTessVertices);
		}
	}
};

class LagrangeCurve : public Curve {
	vector<float> ts;
	float L(int i, float t) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++) {
			if (j != i) { Li *= (t - ts[j]) / (ts[i] - ts[j]); }
		}
		return Li;
	}
public:
	void AddControlPoint(vec2 p) {
		Curve::AddControlPoint(p);
		ts.clear();
		if (cps.size() > 1) {
			float sum = 0.0f;
			vector<float> dsts;
			for (int i = 0; i < cps.size() - 1; ++i) {
				float dst = sqrt(pow(cps[i].x - cps[i + 1].x, 2) + pow(cps[i].y - cps[i + 1].y, 2));
				dsts.push_back(dst);
				sum += dst;
			}
			ts.push_back(0.0f);
			for (int i = 0; i < dsts.size(); ++i) { ts.push_back(ts.back() + dsts[i] / sum); }
		}
	}
	float tStart() { return ts[0]; }
	float tEnd() { return ts[cps.size() - 1]; }
	vec2 r(float t) {
		vec2 rt(0, 0);
		for (int i = 0; i < cps.size(); i++) { rt = rt + cps[i] * L(i, t); }
		return rt;
	}
};

class BezierCurve : public Curve {
	float B(int i, float t) {
		int n = cps.size() - 1;
		float c = 1;
		for (int j = 1; j <= i; j++) { c *= (float)(n - j + 1) / j; }
		return c * pow(t, i) * pow(1 - t, n - i);
	}
public:
	float tStart() { return 0; }
	float tEnd() { return 1; }
	vec2 r(float t) {
		vec2 rt(0, 0);
		for (int i = 0; i < cps.size(); i++) { rt = rt + cps[i] * B(i, t); }
		return rt;
	}
};

float tension = 0;

class CatmullRom : public Curve {
	vector<float> ts;
	vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		t -= t0;
		float deltaT = t1 - t0;
		float deltaT2 = deltaT * deltaT;
		float deltaT3 = deltaT * deltaT2;
		vec2 a2 = (p1 - p0) * 3 / deltaT2 - (v1 + v0 * 2) / deltaT;
		vec2 a3 = (p0 - p1) * 2 / deltaT3 + (v1 + v0) / deltaT2;
		return ((a3 * t + a2) * t + v0) * t + p0;
	}
public:
	void AddControlPoint(vec2 p) {
		Curve::AddControlPoint(p);
		ts.clear();
		if (cps.size() > 1) {
			float sum = 0.0f;
			vector<float> dsts;
			for (int i = 0; i < cps.size() - 1; ++i) {
				float dst = sqrt(pow(cps[i].x - cps[i + 1].x, 2) + pow(cps[i].y - cps[i + 1].y, 2));
				dsts.push_back(dst);
				sum += dst;
			}
			ts.push_back(0.0f);
			for (int i = 0; i < dsts.size(); ++i) {	ts.push_back(ts.back() + dsts[i] / sum); }
		}
	}
	float tStart() { return ts[0]; }
	float tEnd() { return ts[cps.size() - 1]; }
	vec2 r(float t) {
		for (int i = 0; i < cps.size() - 1; i++) {
			if (ts[i] <= t && t <= ts[i + 1]) {
				vec2 vPrev = (i > 0) ? (cps[i] - cps[i - 1]) * (1.0f / (ts[i] - ts[i - 1])) : vec2(0, 0);
				vec2 vCur = (cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]);
				vec2 vNext = (i < cps.size() - 2) ? (cps[i + 2] - cps[i + 1]) / (ts[i + 2] - ts[i + 1]) : vec2(0, 0);
				vec2 v0 = (1 - tension) / 2 * (vPrev + vCur);
				vec2 v1 = (1 - tension) / 2 * (vCur + vNext);
				return Hermite(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t);
			}
		}
		return cps[0];
	}
};

Curve* curve;

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
		case('b'): delete curve;
			curve = new BezierCurve();
			break;
		case('l'): delete curve;
			curve = new LagrangeCurve();
		    break;
		case('c'): delete curve;
			curve = new CatmullRom();
		    break;
		case('Z'): camera.Zoom(1.1f);
			break;
		case('z'): camera.Zoom(1.0f / 1.1f);
			break;
		case('P'): camera.Pan(vec2(1.0f, 0));
			break;
		case('p'): camera.Pan(vec2(-1.0f, 0));
			break;
		case('T'): tension += 0.1f;
			break;
		case('t'): tension -= 0.1f;
			break;
		default: break;
	}
	glutPostRedisplay();
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glPointSize(10);
	glLineWidth(2);

	curve = new LagrangeCurve();

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	curve->Draw();
	glutSwapBuffers();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

int pickedControlPoint = -1;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		curve->AddControlPoint(vec2(cX, cY));
		glutPostRedisplay();
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		pickedControlPoint = curve->PickControlPoint(vec2(cX, cY));
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (pickedControlPoint >= 0) curve->MoveControlPoint(pickedControlPoint, vec2(cX, cY));
	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}