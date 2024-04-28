#include "framework.h"

enum MaterialType { ROUGH, REFLECTIVE, REFRACTIVE };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	vec3 nr = 1;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : public Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH)
	{
		ka = _kd * 3;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}

};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : public Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE)
	{
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct RefractiveMaterial : public Material {
	RefractiveMaterial(vec3 n, vec3 kappa) : Material(REFRACTIVE)
	{
		nr = n;
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
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
	Ray(vec3 _start, vec3 _dir, bool in) { start = _start; dir = normalize(_dir); out = in; }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Plane : public Intersectable {
	vec3 normal;
	vec3 origin;
	float length;
	Material* primaryMaterial;
	Material* secondaryMaterial; // második anyag a sakktáblához

	float planeConstant;

	float minX, maxX;
	float minY, maxY;
	float minZ, maxZ;

	Plane(const vec3& n, const vec3& o, float len, Material* mat1, Material* mat2) {
		normal = normalize(n);
		origin = o;
		length = len;
		primaryMaterial = mat1;
		secondaryMaterial = mat2;

		planeConstant = -dot(origin, normal);

		float halfLength = length / 2;

		minX = origin.x - halfLength;
		maxX = origin.x + halfLength;

		minY = origin.y - halfLength;
		maxY = origin.y + halfLength;

		minZ = origin.z - halfLength;
		maxZ = origin.z + halfLength;
	}

	Hit intersect(const Ray& ray) override {
		Hit hitResult;

		vec3 rayStart = ray.start;
		vec3 rayDir = ray.dir;

		hitResult.t = -(dot(normal, rayStart) + planeConstant) / dot(normal, rayDir);

		if (hitResult.t < 0) {
			return Hit(); // no hit
		}

		hitResult.position = rayStart + rayDir * hitResult.t;

		vec3 intersectionPos = hitResult.position;
		if (intersectionPos.x >= minX && intersectionPos.x <= maxX &&
			intersectionPos.y >= minY && intersectionPos.y <= maxY &&
			intersectionPos.z >= minZ && intersectionPos.z <= maxZ) {

			float tileSize = length / 20; // size of the checkerboard tile

			// Calculate the checkerboard index for each axis
			int xTileIndex = static_cast<int>((intersectionPos.x - minX) / tileSize);
			int yTileIndex = static_cast<int>((intersectionPos.y - minY) / tileSize);
			int zTileIndex = static_cast<int>((intersectionPos.z - minZ) / tileSize);

			// Determine if it's an even or odd tile
			bool isEvenTile = (xTileIndex + yTileIndex + zTileIndex) % 2 == 0;

			// Assign the appropriate material
			hitResult.material = isEvenTile ? secondaryMaterial : primaryMaterial;

			hitResult.normal = normal;

			return hitResult;
		}
		else {
			return Hit(); // no hit within plane bounds
		}
	}
};

struct Cylinder : public Intersectable {
	vec3 baseCenter;
	vec3 axis;
	float height;
	float radius;
	Material* material;

	Cylinder(const vec3& base, const vec3& axisDirection, float cylHeight, float cylRadius, Material* mat) {
		baseCenter = base;
		axis = normalize(axisDirection); // Ensure the axis is a unit vector
		height = cylHeight;
		radius = cylRadius;
		material = mat;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;
		vec3 rayStart = ray.start;
		vec3 rayDir = ray.dir;
		vec3 originToRay = rayStart - baseCenter;

		// Coefficients for the quadratic equation
		float a = dot(rayDir, rayDir) - pow(dot(rayDir, axis), 2);
		float b = 2 * (dot(rayDir, originToRay) - dot(rayDir, axis) * dot(originToRay, axis));
		float c = dot(originToRay, originToRay) - pow(dot(originToRay, axis), 2) - (radius * radius);

		float discriminant = (b * b) - (4.0f * a * c);
		if (discriminant < 0) {
			return Hit(); // No intersection with the cylinder
		}

		float sqrtDiscriminant = sqrtf(discriminant);
		float t1 = (-b - sqrtDiscriminant) / (2.0f * a);
		float t2 = (-b + sqrtDiscriminant) / (2.0f * a);

		if (t1 < 0 && t2 < 0) {
			return Hit(); // No valid intersection
		}

		hit.t = (t1 > 0) ? t1 : t2;
		hit.position = rayStart + (rayDir * hit.t);

		// Check if the hit position is within the height bounds of the cylinder
		vec3 hitToBase = hit.position - baseCenter;
		float projectionOnAxis = dot(hitToBase, axis);
		if (projectionOnAxis < 0 || projectionOnAxis > height) {
			return Hit(); // Out of bounds
		}

		vec3 perpendicularToAxis = hit.position - (baseCenter + axis * projectionOnAxis);
		hit.normal = normalize(perpendicularToAxis);
		hit.material = material;

		return hit;
	}
};


struct Cone : public Intersectable {
	vec3 p;
	vec3 n;
	float h;
	float alpha;

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

		float cos_2 = (float)cos(alpha) * (float)cos(alpha);

		vec3 z = s - p;

		float a = powf(dot(d, n), 2) - dot(d, d) * cos_2;

		float b = 2 * dot(d, n) * dot(z, n) - 2 * dot(d, z) * cos_2;

		float c = powf(dot(z, n), 2) - dot(z, z) * cos_2;

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
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		this->fov = fov;
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

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
	std::vector<vec3> eyes;
	int position;
public:
	void build() {
		std::vector<vec3> push1= { vec3(0, 1, 4) , vec3(4, 1, 0) , vec3(0, 1, -4) , vec3(-4, 1, 0) };
		std::vector<vec2> push2 = { vec2(1,1), vec2(1,-1), vec2(-1,-1), vec2(-1,1) };
		for (int i = 0; i != 4; i++) {
			eyes.push_back(push1[i]);
			eyes.push_back(vec3(push2[i].x * (2*sqrtf(2)), 1, push2[i].y*(2 * sqrtf(2))));
		}
		
		position = 0;
		change();

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.1f, 0.2f, 0.3f), ks1(2, 2, 2);

		Material* blue = new RoughMaterial(vec3(0, 0.1, 0.3), vec3(2, 2, 2), 100);
		Material* white = new RoughMaterial(vec3(0.3, 0.3, 0.3), vec3(2, 2, 2), 100);
		Material* gold = new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		Material* water = new RefractiveMaterial(vec3(1.3, 1.3, 1.3), vec3(0,0,0));
		Material* yellow = new RoughMaterial(vec3(0.3, 0.2, 0.1), vec3(2, 2, 2), 50);
		Material* cyan = new RoughMaterial(vec3(0.1, 0.2, 0.3), vec3(2, 2, 2), 100);
		Material* magenta = new RoughMaterial(vec3(0.3, 0, 0.2), vec3(2, 2, 2), 20);

		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1, -1, -0.05), 0.2, 2, cyan));
		objects.push_back(new Cone(vec3(0, 1, 0.8), vec3(0.2, -1, 0), 0.2, 2, magenta));
		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1, 1, 0), 2, 0.3, gold));
		objects.push_back(new Cylinder(vec3(0, -1, -0.8), vec3(-0.2, 1, -0.1), 2, 0.3, water));
		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 2, 0.3, yellow));
		objects.push_back(new Plane(vec3(0, 1, 0), vec3(0, -1, 0), 20, blue, white));
	}

	void change() {
		if (position >= eyes.size()) { position = 0; }
		camera.set(eyes[position++], vec3(0, 0, 0), vec3(0, 1, 0), 45 * M_PI / 180);
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
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance = vec3(0, 0, 0);
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		else {
			if (hit.material->type == REFLECTIVE) {
				vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
				float cosa = -dot(ray.dir, hit.normal);
				vec3 one(1, 1, 1);
				vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
				outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
			}else{
				float ior = ray.out ? hit.material->nr.x : 1 / hit.material->nr.x;

				float cosTheta = -dot(hit.normal, ray.dir);

				vec3 refractionDir = refract(ray.dir, hit.normal, ior);

				if (length(refractionDir) > 0) {
					Ray refractRay(hit.position - hit.normal * epsilon, refractionDir, !ray.out);
					vec3 one(1, 1, 1);
					float cosa = -dot(ray.dir, hit.normal);
					vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
					outRadiance = outRadiance + trace(refractRay, depth + 1) * (vec3(1, 1, 1) - F);
				}
			}
		}
		return outRadiance;
	}
	vec3 refract(vec3 v, vec3 n, float nr) {
		float cosTheta = -dot(n, v);
		float disc = 1 - (1 - cosTheta * cosTheta) / nr / nr;
		if (disc < 0) return vec3(0, 0, 0);
		return v / nr + n * (cosTheta / nr - sqrtf(disc));
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
	unsigned int vao, textureId = 0;	// vertex array object id and texture id
	//Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
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

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0)
		{
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);

	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { // Camera move function TO-DO
	if (key == 'a') {
		scene.change();
		std::vector<vec4> image(windowWidth * windowHeight);
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		scene.render(image);
		long timeEnd = glutGet(GLUT_ELAPSED_TIME);
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
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
	//scene.Animate(0.1f);
	glutPostRedisplay();
}