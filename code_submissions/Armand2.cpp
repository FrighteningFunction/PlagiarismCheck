//=============================================================================================
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
// Nev    : Szokoly-Angyal Armand
// Neptun : VN450W
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

#if defined(HF) && HF==2

const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vertexPosition;

	void main() {
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		//the color of the primitive
	out vec4 fragmentColor;		// computed color of the current pixel

	void main() {
		fragmentColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";


class Camera {
	vec2 wCenter;
	vec2 wSize;

public:
	Camera() : wCenter(0, 0), wSize(30, 30) {}

	mat4 V() {
		return TranslateMatrix(-wCenter);
	}

	mat4 P() {
		return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y));
	}

	mat4 Vinv() {
		return TranslateMatrix(wCenter);
	}

	mat4 Pinv() {
		return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2));
	}

	void Zoom(float s) {
		wSize = wSize * s;
	}

	void Pan(float p) {
		wCenter = wCenter + p;
	}
};
//end class Camera

Camera camera;
GPUProgram gpuProgram;

class Curve {
	//a görbét kirajzoló görbe pontjai
	std::vector<float> curveVData;
	//a kontrollpontokat kirajzoló adattömb
	std::vector<float> cpsVData;
	unsigned int vao;
	unsigned int vbo[2];

	void recalculate() {
		kts.clear();

		if (cps.size() > 1) {
			std::vector<float> distances;
			float totalDistance = 0.0f;
			// Calculate distances between successive control points
			for (size_t i = 1; i < cps.size(); ++i) {
				float d = sqrtf(powf(cps[i].x - cps[i - 1].x, 2) + powf(cps[i].y - cps[i - 1].y, 2));
				distances.push_back(d);
				totalDistance += d;
			}

			// Recalculate knot values based on proportional distances
			kts.push_back(0.0f);
			float accumulatedDistance = 0.0f;
			for (float distance : distances) {
				accumulatedDistance += distance;
				kts.push_back(accumulatedDistance / totalDistance);
			}
		}
		else {
			kts.push_back(0.0f);
		}

		//recalculate all the curve
		curveVData.clear();

		//tesselate curve
		for (int i = 0; i < 100; i++) {
			float tNormalized = (float)i / 99;
			float t = tStart() + (tEnd() - tStart()) * tNormalized;
			vec2 p = r(t);

			curveVData.push_back(p.x);
			curveVData.push_back(p.y);

		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, curveVData.size() * sizeof(float), &curveVData[0], GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, cpsVData.size() * sizeof(float), &cpsVData[0], GL_DYNAMIC_DRAW);

	}
protected:
	float tau = 0.0f;

	std::vector<vec2> cps;
	std::vector<float> kts;

public:

	Curve() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, &vbo[0]);
	}

	~Curve() {
		glDeleteBuffers(2, &vbo[0]);
		glDeleteVertexArrays(1, &vao);
	}

	void modifyTau(float val) {
		this->tau += val;
	}

	std::vector<vec2>& getControlPoints() {
		return cps;
	}

	virtual vec2 r(float t) = 0;

	virtual float tEnd() = 0;
	virtual float tStart() = 0;



	void addPoint(float cX, float cY) {
		vec4 mVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		vec2 currentCps = vec2(mVertex.x, mVertex.y);

		cps.push_back(currentCps);

		cpsVData.push_back(mVertex.x);
		cpsVData.push_back(mVertex.y);

		recalculate();

		printf("%3.2f, %3.2f control point was added to Curve.\n", mVertex.x, mVertex.y);
	}

	void refresh() {
		cpsVData.clear();

		for (vec2 cp : cps) {
			cpsVData.push_back(cp.x);
			cpsVData.push_back(cp.y);
		}

		recalculate();
	}

	void Draw() {
		if (!curveVData.empty()) {
			mat4 mvpTransform = camera.V() * camera.P();
			gpuProgram.setUniform(mvpTransform, "MVP");


			//draw curve
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1, 1, 0);

			glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
			glBindVertexArray(vao);
			glEnableVertexAttribArray(0);
			glDrawArrays(GL_LINE_STRIP, 0, curveVData.size() / 2);

			//draw control points
			location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1, 0, 0);

			glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
			glBindVertexArray(vao);
			glEnableVertexAttribArray(0);
			glDrawArrays(GL_POINTS, 0, cpsVData.size() / 2);
		}

	}

};
//end class Curve

class Lagrange : public Curve {
	float L(size_t i, float t) {
		float Li = 1.0f;
		for (size_t j = 0; j < cps.size(); j++)
			if (j != i) {
				Li *= (t - kts[j]) / (kts[i] - kts[j]);
			}
		return Li;
	}
public:
	vec2 r(float t) {
		vec2 rt(0, 0);
		for (size_t i = 0; i < cps.size(); i++)
			rt = rt + cps[i] * L(i, t);

		return rt;
	}

	float tEnd() {
		return kts[cps.size()-1];
	}

	float tStart() {
		return kts[0];
	}
};
//end class Lagrange

class BezierCurve : public Curve {

	float B(int i, float t) {
		int n = cps.size() - 1; // n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * powf(t, i) * powf(1 - t, n - i);
	}

public:

	vec2 r(float t) {
		vec2 rt(0, 0);
		for (size_t i = 0; i < cps.size(); i++) rt = rt + cps[i] * B(i, t);

		return rt;
	}

	float tStart() {
		return 0;
	}

	float tEnd() {
		return 1;
	}

};
//end class Lagrange

class CatmullRom : public Curve {
	vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		float deltat = t1 - t0;
		t -= t0;
		float deltat2 = deltat * deltat;
		float deltat3 = deltat2 * deltat;

		vec2 a0 = p0;
		vec2 a1 = v0;
		vec2 a2 = 3 * (p1 - p0) / deltat2 - (v1 + 2 * v0) / deltat;
		vec2 a3 = 2 * (p0 - p1) / deltat3 + (v1 + v0) / deltat2;

		return ((a3 * t + a2) * t + a1) * t + a0;
	}

public:
	vec2 r(float t) {
		for (size_t i = 0; i < cps.size() - 1; i++) {
			if (kts[i] <= t && t <= kts[i + 1]) {
				vec2 vBefore, vNow, vAfter;
				vec2 v0, v1;

				vNow = (cps[i + 1] - cps[i]) / (kts[i + 1] - kts[i]);

				// Calculate v0
				if (i == 0) {
					vBefore = vec2(0, 0);
				}
				else {
					vBefore = (cps[i] - cps[i - 1]) / (kts[i] - kts[i - 1]);
				}
				v0 = 0.5f * (1.0f - tau) * (vBefore + vNow);

				// Calculate v1
				if (i >= cps.size() - 2) {
					vAfter = vec2(0, 0);
				}
				else {
					vAfter = (cps[i + 2] - cps[i + 1]) / (kts[i + 2] - kts[i + 1]);
				}
				v1 = 0.5f * (1.0f - tau) * (vAfter + vNow);

				return Hermite(cps[i], v0, kts[i], cps[i + 1], v1, kts[i + 1], t);
			}
		}

		return cps[0];
	}

	float tStart() {
		return kts[0];
	}

	float tEnd() {
		return kts[cps.size() - 1];
	}
};

//end class CatmullRom

//--------------- GLOBAL OBJECTS -----------------

bool selection = false;

Curve* current;

vec2* chosen = NULL;

//--------------- EVENT HANDLERS -----------------



void onInitialization() {

	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f); // Width of lines in pixels

	glPointSize(10.0f); //Width of point in pixels

	current = new Lagrange();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0.502, 0.502, 0.502, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	current->Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {

	switch (key) {
	case 'l':
		delete current;
		current = new Lagrange();
		printf("selected LAGRANGE\n");
		break;
	case 'b':
		delete current;
		current = new BezierCurve();
		printf("selected BEZIER\n");
		break;
	case 'c':
		delete current;
		current = new CatmullRom();
		printf("selected CATMULL-ROM\n");
		break;
	case 'Z':
		camera.Zoom(1.1f);
		break;
	case 'z':
		camera.Zoom(1.0f / 1.1f);
		break;
	case 'P':
		camera.Pan(1);
		break;
	case 'p':
		camera.Pan(-1);
		break;
	case 'T':
		current->modifyTau(0.1f);
		current->refresh();
		printf("tau modified by +0.1\n");
		break;
	case 't':
		current->modifyTau(-0.1f);
		current->refresh();
		printf("tau modified by -0.1\n");
		break;
	default:
		printf("invalid key pressed.\n");
		break;
	}
	glutPostRedisplay();
}


void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (selection) {
		vec2 ref = *chosen;
		vec4 mVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		*chosen = vec2(mVertex.x, mVertex.y);
		printf("ONMOUSEMOTION: %3.2f, %3.2f control point was set to %3.2f, %3.2f\n", ref.x, ref.y, mVertex.x, mVertex.y);

		current->refresh();
		glutPostRedisplay();
	}
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {

		if (selection) {
			vec3 ref = *chosen;
			vec4 mVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
			*chosen = vec2(mVertex.x, mVertex.y);
			printf("%3.2f, %3.2f control point was set to %3.2f, %3.2f\n", ref.x, ref.y, mVertex.x, mVertex.y);
			selection = false;
			chosen = NULL;
			current->refresh();
		}
		else {
			current->addPoint(cX, cY);
		}

		glutPostRedisplay();
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		vec4 mClick = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		vec2 click = vec2(mClick.x, mClick.y);
		auto& controlPoints = current->getControlPoints();
		for (size_t i = 0; i < controlPoints.size(); ++i) {
			float d = sqrtf(powf(controlPoints[i].x - click.x, 2) + powf(controlPoints[i].y - click.y, 2));
			if (d < 0.1) {
				chosen = &controlPoints[i];
				selection = true;
				printf("%3.2f, %3.2f control point was chosen.\n", controlPoints[i].x, controlPoints[i].y);
				break;
			}
			else {
				chosen = NULL;
				selection = false;
			}
		}
	}

}

void onIdle() {
}

#endif