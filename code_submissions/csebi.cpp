//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Csányi Balázs
// Neptun : BVG0EJ
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

    bool refractive = false;
    bool reflective = false;
    vec3 reflectionNumber = 1;


    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * 3), kd(_kd), ks(_ks) { shininess = _shininess; }


    Material(bool _reflective, bool _refractive, vec3 _kappa, vec3 _n_reflect) : kappa(_kappa), refractive(_refractive), reflective(_reflective), reflectionNumber(_n_reflect) {}
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, direction;

    bool out = true;

    Ray(vec3 _start, vec3 _dir) { start = _start; direction = normalize(_dir); }

    Ray(vec3 _start, vec3 _direction, bool _out) { start = _start; direction = normalize(_direction); out = _out; }
};

class Intersectable {
protected:
	Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};




struct Plane : public Intersectable {
    vec3 orientation;
    vec3 origin;
    float size;
    Material* material1;
    Material* material2;
    float distance;
    float left, right;
    float bottom, top;
    float back, front;

    Plane(const vec3& _orientation, const vec3& _origin, const float& _size, Material* _material1, Material* _material2) {
        orientation = normalize(_orientation);
        origin = _origin;
        size = _size;
        material1 = _material1;
        material2 = _material2;

        distance = -dot(origin, orientation);

        float halfSize = size / 2;

        left = origin.x - halfSize;
        right = origin.x + halfSize;

        bottom = origin.y - halfSize;
        top = origin.y + halfSize;

        back = origin.z - halfSize;
        front = origin.z + halfSize;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;

        vec3 s = ray.start;
        vec3 d = ray.direction;

        hit.t = -(dot( orientation, s) + distance) / dot(orientation, d);

        if (hit.t < 0) {
            return Hit();
        }

        hit.position = s + d * hit.t;

		vec3 ps = hit.position;

		if (ps.x >= left && ps.x <= right &&
			ps.y >= bottom && ps.y <= top &&
			ps.z >= back && ps.z <= front) {
			float cellSize = size / 20;


			float xOffset = 0.5 - (cellSize/2);
			float yOffset = 0- (cellSize/2);
			float zOffset = -0.5 - (cellSize/2);


			int xIndex = static_cast<int>((ps.x - left - xOffset) / cellSize);
			int yIndex = static_cast<int>((ps.y - bottom - yOffset) / cellSize);
			int zIndex = static_cast<int>((ps.z - back - zOffset) / cellSize);

			bool even = (xIndex + yIndex + zIndex) % 2 == 0;

			hit.material = even ? material1 : material2;

            hit.normal = orientation;

            return hit;
        }
        else {
            return Hit();
        }
    }
};


struct Cylinder : public Intersectable {
    vec3 point;
    vec3 direction;
    float length;
    float radius;

    Cylinder(const vec3& _anchor, const vec3& _dir, float _len, float _R, Material* _material) {
        point = _anchor;
        direction = normalize(_dir);
        radius = _R;
        length = _len;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 s = ray.start;
        vec3 d = ray.direction;
        vec3 offset = s - point;

        float a = dot(d, d) - 2 * powf(dot(d, direction), 2) + powf(dot(d, direction), 2) * dot(direction, direction);
        float b = 2 * dot(d, offset) - 4 * dot(d, direction) * dot(direction, offset) + 2 * dot(d, direction) * dot(direction, direction) * dot(direction, offset);
        float c = -radius * radius - 2 * powf(dot(direction, offset), 2) + dot(direction, direction) * powf(dot(direction, offset), 2) + dot(offset, offset);

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0) return hit;

        float sqrt_discriminant = sqrtf(discriminant);
        float t1 = (-b + sqrt_discriminant) / 2.0f / a;
        float t2 = (-b - sqrt_discriminant) / 2.0f / a;

        if (t1 <= 0 && t2 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = s + d * hit.t;

        if (dot(hit.position - point, direction) > length || dot(hit.position - point, direction) < 0) {
            return Hit();
        }

        vec3 diff = hit.position - point;
		hit.normal = normalize(diff - direction * dot(diff, direction));
        hit.material = material;
        return hit;
    }
};

struct Cone : public Intersectable {
    vec3 apex;
    vec3 orientation;
    float length;
    float angle;

    Cone(const vec3& _apex, vec3 _dir, float _angle, float _len, Material* _material) {
        apex = _apex;
        orientation = normalize(_dir);
        angle = _angle;
        length = _len;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 start = ray.start;
        vec3 rayDir = ray.direction;

        float cosSquared = cos(angle) * cos(angle);

        vec3 offset = start - apex;

        float a = powf(dot(rayDir, orientation), 2) - dot(rayDir, rayDir) * cosSquared;
        float b = 2 * dot(rayDir, orientation) * dot(offset, orientation) - 2 * dot(rayDir, offset) * cosSquared;
        float c = powf(dot(offset, orientation), 2) - dot(offset, offset) * cosSquared;

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0) return hit;

        float sqrtDiscriminant = sqrtf(discriminant);
        float t0 = (-b + sqrtDiscriminant) / 2.0f / a;
        float t1 = (-b - sqrtDiscriminant) / 2.0f / a;

        if (t0 <= 0 && t1 <= 0) return hit;
        hit.t = (t1 > 0) ? t0 : t1;
        hit.position = start + rayDir * hit.t;

        if (dot(hit.position - apex, orientation) > length || dot(hit.position - apex, orientation) < 0) {
            return Hit();
        }

        hit.normal = normalize(2 * dot(hit.position - apex, orientation) * orientation - 2 * (hit.position - apex) * cosSquared);
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
		vec3 direction = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, direction);
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
    vec3 Ambient;



    std::vector<vec3> POVs;
    int view;

	const int maxDepth = 20;
public:


    void build() {


        POVs.push_back(vec3(0, 1, 4));
        POVs.push_back(vec3(2 * sqrtf(2), 1, 2 * sqrtf(2)));
        POVs.push_back(vec3(4, 1, 0));
        POVs.push_back(vec3(2 * sqrtf(2), 1, -(2 * sqrtf(2))));
        POVs.push_back(vec3(0, 1, -4));
        POVs.push_back(vec3(-(2 * sqrtf(2)), 1, -(2 * sqrtf(2))));
        POVs.push_back(vec3(-4, 1, 0));
        POVs.push_back(vec3(-(2 * sqrtf(2)), 1, 2 * sqrtf(2)));
        view = 0;
        nextPOV();



        Ambient = vec3(0.4f, 0.4f, 0.4f);
        vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        vec3 kd1(0.0f, 0.0f, 0.0f);
        vec3 ks1(2, 2, 2);

        kd1 = vec3(0, 0.1f, 0.3f);
        Material* blueTile = new Material(kd1, ks1, 100);

        kd1 = vec3(0.3f, 0.3f, 0.3f);
        Material* whiteTile = new Material(kd1, ks1, 100);

        kd1 = vec3(0.1f, 0.2f, 0.3f);
        Material* cyan = new Material(kd1, ks1, 100);

        kd1 = vec3(0.3f, 0.2f, 0.1f);
        Material* plastic = new Material(kd1, ks1, 50);

        kd1 = vec3(0.3f, 0, 0.2f);
        Material* magenta = new Material(kd1, ks1, 20);
        
        vec3 kappa = vec3(3.1f, 2.7f, 1.9f);
		vec3 nr = vec3(0.17f, 0.35f, 1.5f);
        Material* gold = new Material(true, false, kappa, nr);

		//víz
		kappa = vec3(0, 0, 0);
		nr = vec3(1, 1, 1) * 1.3f;
		Material* water = new  Material(true, true, kappa, nr);

        objects.push_back(new Plane(vec3(0, 1, 0), vec3(0, -1, 0), 20, blueTile, whiteTile));
		//cones
		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1f, -1, -0.05f), 0.2f, 2, cyan));
		objects.push_back(new Cone(vec3(0, 1, 0.8f), vec3(0.2f, -1, 0), 0.2f, 2, magenta));
		//plastic cylinder
		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1f), 2, 0.3f, plastic));
		//golden cylinder
		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1f, 1, 0), 2, 0.3f, gold));
		//water cylinder
		objects.push_back(new Cylinder(vec3(0, -1, -0.8f), vec3(-0.2f, 1, -0.1f), 2, 0.3f, water));
    }

    void nextPOV() {
        if (view >= POVs.size()) {
            view = 0;
        }
        camera.set(POVs[view++], vec3(0, 0, 0), vec3(0, 1, 0), 45 * M_PI / 180);
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

    vec3 fresnel(vec3 normal, vec3 direction, const Material& mat) {
        float cosTheta = -dot(normal, direction);
        vec3 oneVec = vec3(1, 1, 1);

        vec3 nminus1_2 = (mat.reflectionNumber - oneVec) * (mat.reflectionNumber - oneVec);
        vec3 nplus1_2 = (mat.reflectionNumber + oneVec) * (mat.reflectionNumber + oneVec);

        vec3 f0 = divide(nminus1_2 + mat.kappa * mat.kappa, nplus1_2 + mat.kappa * mat.kappa);

        return f0 + (oneVec - f0) * powf(1 - cosTheta, 5);
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > maxDepth) return Ambient;

        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return Ambient;

        vec3 outRadiance = vec3(0, 0, 0);

        const Material mat = *hit.material;

        if (!mat.reflective && !mat.refractive) {
            outRadiance = hit.material->ka * Ambient;

            for (Light* light : lights) {
                Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
                float cosTheta = dot(hit.normal, light->direction);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                    outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.direction + light->direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
            }
        }

        if (mat.reflective) {
            float cosTheta = - dot(hit.normal, ray.direction);

            vec3 reflectionDir = ray.direction + 2 * hit.normal * cosTheta;

            Ray reflectRay = Ray(hit.position + epsilon*hit.normal, reflectionDir, ray.out);

            outRadiance = outRadiance + trace(reflectRay, depth + 1) * fresnel(hit.normal, ray.direction, mat);
        }

        if (mat.refractive) {

            float ior = ray.out ? mat.reflectionNumber.x : 1 / mat.reflectionNumber.x;

            vec3 refractionDir = refract(ray.direction, hit.normal, ior);

            if (length(refractionDir) > 0) {
                Ray refractRay(hit.position - hit.normal * epsilon, refractionDir, !ray.out);

                outRadiance = outRadiance + trace(refractRay, depth + 1) * (vec3(1, 1, 1) - fresnel(hit.normal, ray.direction, mat));
            }
        }

        return outRadiance;
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
        if (dot(ray.direction, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }


};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
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
const char *fragmentSource = R"(
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

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { // Camera move function TO-DO
    if (key == 'a') {
        scene.nextPOV();
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
