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

struct Material {
	vec3 ka, kd, ks;
	float  shininess;

	vec3 kappa;

	//ha mindkettõ false, akkor rough
	bool refractive = false;
	bool reflective = false;

	vec3 nr = 1;

	//rough anyagoknak
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * 3), kd(_kd), ks(_ks) { shininess = _shininess; }

	//reflective vagy refractive anyagoknak
	Material(bool _reflective, bool _refractive, vec3 _kappa, vec3 _n_reflect) : kappa(_kappa), refractive(_refractive), reflective(_reflective), nr(_n_reflect) {}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;

	bool out = true;

	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }

	Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out = _out; }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Plane : public Intersectable {
	vec3 n;
	vec3 o;
	float l;
	Material* mat1;
	Material* mat2;

	float p;

	float x1, x2;
	float y1, y2;
	float z1, z2;

	Plane(const vec3& _n, const vec3& _o, const float& _l, Material* _mat1, Material* _mat2) {
		n = normalize(_n);
		o = _o;
		l = _l;
		mat1 = _mat1;
		mat2 = _mat2;

		p = -dot(o, n);

		float half = l / 2;

		x1 = o.x - half;
		x2 = o.x + half;

		y1 = o.y - half;
		y2 = o.y + half;

		z1 = o.z - half;
		z2 = o.z + half;
	}

	Hit intersect(const Ray& incomingRay) {
		Hit resultHit;

		vec3 rayStart = incomingRay.start;
		vec3 rayDirection = incomingRay.dir;

		resultHit.t = -((dot(n, rayStart) + p) / dot(n, rayDirection));

		if (resultHit.t < 0) {
			return Hit();
		}

		resultHit.position = rayStart + rayDirection * resultHit.t;
		vec3 position = resultHit.position;

		if (isWithinBounds(position)) {
			vec3 offset = calculateOffset();
			float divisionSize = l / 20;

			int xIdx = static_cast<int>((position.x - x1 - offset.x) / divisionSize);
			int yIdx = static_cast<int>((position.y - y1 - offset.y) / divisionSize);
			int zIdx = static_cast<int>((position.z - z1 - offset.z) / divisionSize);

			bool isEven = (xIdx + yIdx + zIdx) % 2 == 0;

			resultHit.material = isEven ? mat1 : mat2;
			resultHit.normal = n;

			return resultHit;
		}
		else {
			return Hit();
		}
	}

	bool isWithinBounds(const vec3& pos) {
		return pos.x >= x1 && pos.x <= x2 &&
			pos.y >= y1 && pos.y <= y2 &&
			pos.z >= z1 && pos.z <= z2;
	}

	vec3 calculateOffset() {
		float squareSize = l / 20;
		float halfSquare = 0.5 - (squareSize / 2);
		return vec3(halfSquare, halfSquare, -1 - (squareSize / 2));
	}
};

struct Cylinder : public Intersectable {
	vec3 p;
	vec3 v0;
	float h;
	float R;

	Cylinder(const vec3& _p, const vec3& _v0, float _h, float _R, Material* _material) {
		p = _p;
		v0 = normalize(_v0);
		R = _R;
		h = _h;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 s = ray.start;
		vec3 d = ray.dir;
		vec3 z = s - p;
		
		float a = dot(d, d) - 2 * powf(dot(d, v0), 2) + powf(dot(d, v0), 2) * dot(v0, v0);
		float b = 2 * dot(d, z) - 4 * dot(d, v0) * dot(v0, z) + 2 * dot(d, v0) * dot(v0, v0) * dot(v0, z);
		float c = -R * R - 2 * powf(dot(v0, z), 2) + dot(v0, v0) * powf(dot(v0, z), 2) + dot(z, z);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0 && t2 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = s + d * hit.t;

		if (dot(hit.position - p, v0) > h || dot(hit.position - p, v0) < 0) {
			return Hit();
		}

		vec3 diff = hit.position - p;
		hit.normal = normalize(diff - v0 * dot(diff, v0));
		hit.material = material;
		return hit;
	}
};

struct Cone : public Intersectable {
	vec3 p;
	vec3 n;
	float h;
	float alpha;

	/*
	* @param vec3 _p : csúcs
	* @param float _n : tengelyirány
	* @param float _alpha : nyílásszög radiánban
	* @param float _h : magasság
	* @param float _material : anyag
	*/
	Cone(const vec3& _p, vec3 _n, float _alpha, float _h, Material* _material) {
		p = _p;
		n = normalize(_n);
		alpha = _alpha;
		h = _h;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 s = ray.start;
		vec3 d = ray.dir;

		float cos_2 = (float) cos(alpha) * (float) cos(alpha);

		vec3 z = s - p;

		float a = powf(dot(d, n),2) - dot(d, d) * cos_2;

		float b = 2 * dot(d, n) * dot(z, n) - 2 * dot(d, z) * cos_2;

		float c = powf(dot(z, n),2) - dot(z, z) * cos_2;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0 && t2 <= 0) return hit;
		hit.t = (t2 > 0) ? t1 : t2;
		hit.position = s + d * hit.t;

		if (dot(hit.position - p, n) > h || dot(hit.position - p, n) < 0) {
			return Hit();
		}

		hit.normal = normalize(2 * dot(hit.position - p, n) * n - 2 * (hit.position - p) * cos_2);
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};


//EPSILON <<<<--------------------------------------------------------------------------------------------------
const float epsilon = 0.0001f;

//******************************************************************************************
//******************         SCENE RENDER        *******************************************
//******************************************************************************************

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;

	std::vector<vec3> viewPositions;
	int it;

	const int maxDepth = 20;
public:

	Camera& getCamera(){
		return camera;
	}

	void build() {
		float sqrtValue = 2 * sqrtf(2);

		viewPositions.push_back(vec3(0, 1, 4));
		viewPositions.push_back(vec3(sqrtValue, 1, sqrtValue));
		viewPositions.push_back(vec3(4, 1, 0));
		viewPositions.push_back(vec3(sqrtValue, 1, -sqrtValue));
		viewPositions.push_back(vec3(0, 1, -4));
		viewPositions.push_back(vec3(-sqrtValue, 1, -sqrtValue));
		viewPositions.push_back(vec3(-4, 1, 0));
		viewPositions.push_back(vec3(-sqrtValue, 1, sqrtValue));
		it = 0;
		rotateCamera();


		vec3 eye = vec3(0, 1, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.1f, 0.2f, 0.3f), ks1(2, 2, 2);

		//kék sakk mezõ anyaga
		kd1 = vec3(0, 0.1f, 0.3f);
		ks1 = vec3(2, 2, 2);
		Material* bluechess = new Material(kd1, ks1, 100);

		//fehér sakk mezõ anyaga
		kd1 = vec3(0.3f, 0.3f, 0.3f);
		ks1 = vec3(2, 2, 2);
		Material* whitechess = new Material(kd1, ks1, 100);


		//cián kúp anyaga
		kd1 = vec3(0.1f, 0.2f, 0.3f);
		ks1 = vec3(2, 2, 2);
		Material* cian = new Material(kd1, ks1, 100);

		//barna mûanyag
		kd1 = vec3(0.3f, 0.2f, 0.1f);
		ks1 = vec3(2, 2, 2);
		Material* plastic = new Material(kd1, ks1, 50);

		//magenta diffúz-spekuláris
		kd1 = vec3(0.3f, 0, 0.2f);
		ks1 = vec3(2, 2, 2);
		Material* magenta = new Material(kd1, ks1, 20);

		//arany
		vec3 kappa = vec3(3.1f, 2.7f, 1.9f);
		vec3 nr = vec3(0.17f, 0.35f, 1.5f);

		Material* gold = new Material(true, false, kappa, nr);

		//víz
		kappa = vec3(0, 0, 0);
		nr = vec3(1, 1, 1) * 1.3f;

		Material* water = new  Material(true, true, kappa, nr);


		//chess plane
		objects.push_back(new Plane(vec3(0, 1, 0), vec3(0, -1, 0), 20, bluechess, whitechess));
		//cones
		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1f, -1, -0.05f), 0.2f, 2, cian));
		objects.push_back(new Cone(vec3(0, 1, 0.8f), vec3(0.2f, -1, 0), 0.2f, 2, magenta));
		//plastic cylinder
		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1f), 2, 0.3f, plastic));
		//golden cylinder
		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1f, 1, 0), 2, 0.3f, gold));
		//water cylinder
		objects.push_back(new Cylinder(vec3(0, -1, -0.8f), vec3(-0.2f, 1, -0.1f), 2, 0.3f, water));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 divide(vec3 a, vec3 b) {
		return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
	}

	vec3 refract(vec3 v, vec3 n, float nr) {
		float cosTheta = -dot(n, v);

		float disc = 1 - (1 - cosTheta * cosTheta) / nr / nr;

		if (disc < 0) return vec3(0, 0, 0);

		return v / nr + n * (cosTheta / nr - sqrtf(disc));
	}

	vec3 fresnel(const float& cosTheta, const Material& mat) {
		vec3 oneVec = vec3(1, 1, 1);

		vec3 nminus1_2 = (mat.nr - oneVec) * (mat.nr - oneVec);
		vec3 nplus1_2 = (mat.nr + oneVec) * (mat.nr + oneVec);

		vec3 f0 = divide(nminus1_2 + mat.kappa * mat.kappa, nplus1_2 + mat.kappa * mat.kappa);

		return f0 + (oneVec - f0) * powf(1 - cosTheta, 5);
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > maxDepth) return La;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance = vec3(0, 0, 0);

		const Material mat = *hit.material;

		if (!mat.reflective && !mat.refractive) {
			outRadiance = hit.material->ka * La;

			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}

		if (mat.reflective) {
			float cosTheta = - dot(hit.normal, ray.dir);

			vec3 reflectionDir = ray.dir + 2 * hit.normal * cosTheta;

			Ray reflectRay = Ray(hit.position + epsilon*hit.normal, reflectionDir, ray.out);	
			
			outRadiance = outRadiance + trace(reflectRay, depth + 1) * fresnel(cosTheta, mat);
		}

		if (mat.refractive) {

			float ior = ray.out ? mat.nr.x : 1 / mat.nr.x;

			float cosTheta = -dot(hit.normal, ray.dir);

			vec3 refractionDir = refract(ray.dir, hit.normal, ior);

			if (length(refractionDir) > 0) {
				Ray refractRay(hit.position - hit.normal * epsilon, refractionDir, !ray.out);

				outRadiance = outRadiance + trace(refractRay, depth + 1) * (vec3(1, 1, 1) - fresnel(cosTheta, mat));
			}
		}

		return outRadiance;
	}

	void rotateCamera() {
		if (it >= viewPositions.size()) { it = 0; }
		camera.set(viewPositions[it++], vec3(0, 0, 0), vec3(0, 1, 0), 45.0f * (float) M_PI / 180.0f);
	}
};

GPUProgram gpuProgram;
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	

		
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	  
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);    
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}


void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a') {
		printf("pressed a\n");
		scene.rotateCamera();

		std::vector<vec4> image(windowWidth * windowHeight);
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		scene.render(image);
		long timeEnd = glutGet(GLUT_ELAPSED_TIME);
		printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
		glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
//empty
}


void onMouse(int button, int state, int pX, int pY) {
	//empty
}

void onMouseMotion(int pX, int pY) {
	//empty
}

void onIdle() {
	//empty
}