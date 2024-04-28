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

struct Material {
	vec3 ka, kd, ks, k, n;
	float  shininess;
	bool specular, dioptric;

	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * 3), kd(_kd), ks(_ks) {
		k = 0;
		n = 1;
		shininess = _shininess;
		specular = false;
		dioptric = false;
	}

	Material(bool _specular, bool _dioptric, vec3 _n, vec3 _k) : k(_k), n(_n), specular(_specular), dioptric(_dioptric) {
		ka = 0;
		kd = 0;
		ks = 0;
		shininess = 0;
	}
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

struct ChequeredSquare : public Intersectable {
	vec3 n, o;
	float l, p;
	Material* material2;

	ChequeredSquare(vec3 _n, vec3 _o, float _l, Material* _material, Material* _material2) :
		n(normalize(_n)), o(_o), l(_l), material2(_material2) {
		p = -dot(o, n);
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit mark;
		float t = -(dot(n, ray.start) + p) / dot(n, ray.dir);
		if (t < 0) return Hit();

		vec3 pos = ray.start + ray.dir * t;
		if (isInsideSquare(pos)) {
			mark.t = t;
			mark.position = pos;
			mark.material = isEvenSquare(pos) ? material : material2;
			mark.normal = n;
			return mark;
		}
		return Hit();
	}

private:
	bool isInsideSquare(vec3 pos) {
		return pos.x >= (o.x - l / 2) && pos.x <= (o.x + l / 2) &&
			pos.y >= (o.y - l / 2) && pos.y <= (o.y + l / 2) &&
			pos.z >= (o.z - l / 2) && pos.z <= (o.z + l / 2);
	}

	bool isEvenSquare(vec3 pos) {
		float side = l / 20;
		int X = static_cast<int>((pos.x - (o.x - l/2) - (0.5 - side / 2)) / side);
		int Y = static_cast<int>((pos.y - (o.y - l/2) - (0.5 - side / 2)) / side);
		int Z = static_cast<int>((pos.z - (o.z - l/2) - (-1 - side / 2)) / side);
		return (X + Y + Z) % 2 == 0;
	}
};

struct Cone : public Intersectable {
	vec3 p, n;
	float h, a;

	Cone(vec3 _p, vec3 _n, float _a, float _h, Material* _material) :
		p(_p), n(normalize(_n)), h(_h), a(_a) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit mark;
		vec3 z = ray.start - p;
		float cos_2 = cos(a) * cos(a);
		float a = dot(ray.dir, n) * dot(ray.dir, n) - dot(ray.dir, ray.dir) * cos_2;
		float b = 2 * dot(ray.dir, n) * dot(z, n) - 2 * dot(ray.dir, z) * cos_2;
		float c = dot(z, n) * dot(z, n) - dot(z, z) * cos_2;
		float dis = b * b - 4.0f * a * c;
		if (dis < 0) return mark;

		float t1 = (-b + sqrtf(dis)) / (2.0f * a);
		float t2 = (-b - sqrtf(dis)) / (2.0f * a);
		if (t1 <= 0 && t2 <= 0) return mark;

		float t = (t2 > 0) ? t1 : t2;
		vec3 pos = ray.start + ray.dir * t;
		if (dot(pos - p, n) > h || dot(pos - p, n) < 0) return mark;

		mark.t = t;
		mark.position = pos;
		mark.normal = normalize(2 * dot(pos - p, n) * n - 2 * (pos - p) * cos_2);
		mark.material = material;
		return mark;
	}
};

struct Cylinder : public Intersectable {
	vec3 p, v;
	float h, r;

	Cylinder(vec3 _p, vec3 _v, float _r, float _h, Material* _material) :
		p(_p), v(normalize(_v)), h(_h), r(_r) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit mark;
		vec3 z = ray.start - p;
		float a = dot(ray.dir, ray.dir) - 2 * powf(dot(ray.dir, v), 2) + powf(dot(ray.dir, v), 2) * dot(v, v);
		float b = 2 * dot(ray.dir, z) - 4 * dot(ray.dir, v) * dot(v, z) + 2 * dot(ray.dir, v) * dot(v, v) * dot(v, z);
		float c = -r * r - 2 * powf(dot(v, z), 2) + dot(v, v) * powf(dot(v, z), 2) + dot(z, z);
		float dis = b * b - 4.0f * a * c;
		if (dis < 0) return mark;

		float t1 = (-b + sqrtf(dis)) / 2.0f / a;
		float t2 = (-b - sqrtf(dis)) / 2.0f / a;
		if (t1 <= 0 && t2 <= 0) return mark;

		mark.t = (t2 > 0) ? t2 : t1;
		mark.position = ray.start + ray.dir * mark.t;
		if (dot(mark.position - p, v) > h || dot(mark.position - p, v) < 0) return Hit();

		mark.normal = normalize(mark.position - p - v * dot(mark.position - p, v));
		mark.material = material;
		return mark;
	}
};

class Camera {
	vec3 e, l, r, u;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		e = _eye;
		l = _lookat;
		vec3 w = e - l;
		float focus = length(w);
		r = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		u = normalize(cross(w, r)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = l + r * (2.0f * (X + 0.5f) / windowWidth - 1) + u * (2.0f * (Y + 0.5f) / windowHeight - 1) - e;
		return Ray(e, dir);
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

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
	std::vector<vec3> eyes;
	unsigned iterator;
public:
	void build() {
		// Camera setup
		eyes.push_back(vec3(0, 1, 4));
		eyes.push_back(vec3(2 * sqrtf(2), 1, 2 * sqrtf(2)));
		eyes.push_back(vec3(4, 1, 0));
		eyes.push_back(vec3(2 * sqrtf(2), 1, -(2 * sqrtf(2))));
		eyes.push_back(vec3(0, 1, -4));
		eyes.push_back(vec3(-(2 * sqrtf(2)), 1, -(2 * sqrtf(2))));
		eyes.push_back(vec3(-4, 1, 0));
		eyes.push_back(vec3(-(2 * sqrtf(2)), 1, 2 * sqrtf(2)));
		iterator = 0;
		setCam();

		// Lights setup
		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		// Objects setup
		vec3 ks(2, 2, 2);
		float s = 100;
		Material* blue = new Material(vec3(0.0, 0.1, 0.3), ks, s);
		Material* white = new Material(vec3(0.3, 0.3, 0.3), ks, s);
		Material* cyan = new Material(vec3(0.1, 0.2, 0.3), ks, s);
		Material* magenta = new Material(vec3(0.3, 0, 0.2), ks, 20);
		Material* yellow = new Material(vec3(0.3, 0.2, 0.1), ks, 50);
		Material* gold = new Material(true, false, vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		Material* water = new Material(true, true, vec3(1.3, 1.3, 1.3), vec3(0, 0, 0));
		objects.push_back(new ChequeredSquare(vec3(0, 1, 0), vec3(0, -1, 0), 20, blue, white));
		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1, -1, -0.05), 0.2, 2, cyan));
		objects.push_back(new Cone(vec3(0, 1, 0.8), vec3(0.2, -1, 0), 0.2, 2, magenta));
		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 0.3, 2, yellow));
		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1, 1, 0), 0.3, 2, gold));
		objects.push_back(new Cylinder(vec3(0, -1, -0.8), vec3(-0.2, 1, -0.1), 0.3, 2, water));
	}

	void setCam() {
		if (iterator >= eyes.size()) { iterator = 0; }
		camera.set(eyes[iterator++], vec3(0, 0, 0), vec3(0, 1, 0), 45 * M_PI / 180);
	}

	void render(std::vector<vec4>& image) {
		for (unsigned Y = 0; Y < windowHeight; Y++) {
			for (unsigned X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit best;
		for (Intersectable* object : objects) {
			Hit mark = object->intersect(ray);
			if (mark.t > 0 && (best.t < 0 || mark.t < best.t))  best = mark;
		}
		if (dot(ray.dir, best.normal) > 0) best.normal = best.normal * (-1);
		return best;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 bend(vec3 v, vec3 n, float nr) {
		float cos = -dot(n, v);
		float dis = 1 - (1 - cos * cos) / nr / nr;
		if (dis < 0) return vec3{ 0, 0, 0 };

		return { v.x / nr + n.x * (cos / nr - sqrtf(dis)),
				v.y / nr + n.y * (cos / nr - sqrtf(dis)),
				v.z / nr + n.z * (cos / nr - sqrtf(dis)) };
	}

	vec3 fres(float cos, Material m) {
		vec3 nm = (m.n - vec3(1, 1, 1)) * (m.n - vec3(1, 1, 1));
		vec3 np = (m.n + vec3(1, 1, 1)) * (m.n + vec3(1, 1, 1));
		vec3 a = nm + m.k * m.k;
		vec3 b = np + m.k * m.k;
		vec3 c = vec3(a.x / b.x, a.y / b.y, a.z / b.z);
		return c + (vec3(1, 1, 1) - c) * powf(1 - cos, 5);
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;

		Hit mark = firstIntersect(ray);
		if (mark.t < 0) return La;

		vec3 out = vec3(0, 0, 0);
		Material m = *mark.material;
		if (!m.specular && !m.dioptric) {
			out = mark.material->ka * La;
			for (Light* light : lights) {
				Ray shadow(mark.position + mark.normal * epsilon, light->direction);
				float cos = dot(mark.normal, light->direction);
				if (cos > 0 && !shadowIntersect(shadow)) {
					out = out + light->Le * mark.material->kd * cos;
					cos = dot(mark.normal, normalize(-ray.dir + light->direction));
					if (cos > 0) out = out + light->Le * mark.material->ks * powf(cos, mark.material->shininess);
				}
			}
		}
		else {
			float cos = -dot(mark.normal, ray.dir);
			if (m.specular) {
				vec3 dir = ray.dir + 2 * mark.normal * cos;
				Ray reflection = Ray(mark.position + epsilon * mark.normal, dir, ray.out);
				out = out + trace(reflection, depth + 1) * fres(cos, m);
			}
			if (m.dioptric) {
				float io = ray.out ? m.n.x : 1 / m.n.x;
				vec3 dir = bend(ray.dir, mark.normal, io);
				if (length(dir) > 0) {
					Ray reflection(mark.position - mark.normal * epsilon, dir, !ray.out);
					out = out + trace(reflection, depth + 1) * (vec3(1, 1, 1) - fres(cos, m));
				}
			}
		}
		return out;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
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
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a') {
		scene.setCam();
		std::vector<vec4> image(windowWidth * windowHeight);
		scene.render(image);
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}