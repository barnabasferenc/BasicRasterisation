#pragma once
#include "pch.h"

using namespace std;
using namespace winrt;
using namespace Windows;
using namespace Windows::Foundation::Numerics;
using namespace DirectX;
using namespace Concurrency;

constexpr float inchToMm = 25.4f;
constexpr int numslices = 50;
constexpr int numstacks = 50;
extern uint32_t ImageWidth;
extern uint32_t ImageHeight;
constexpr float nearClippingPLane = 0.1f;
constexpr float farClippingPLane = 1000.f;
constexpr float focalLength = 35; // in mm // 35mm Full Aperture in inches
constexpr float filmApertureWidth = 0.980f;
constexpr float filmApertureHeight = 0.735f;
constexpr int numtri = (numslices * numstacks + numslices) * 2; //number of triangles of sphere
enum class FitResolutionGate { kFill = 0, kOverscan };

/// <summary>
/// the color format for Direct2D
/// </summary>
struct BGRA
{
    uint8_t b, g, r, a;
    BGRA() : r{ 0 }, g{ 0 }, b{ 0 }, a{ 0 } 
    {}
    BGRA(const BGRA& c) : r{ c.r }, g{ c.g }, b{ c.b }, a{ c.a }
    {}
    BGRA(uint8_t r, uint8_t g, uint8_t b, uint8_t a):r{r}, b{b}, g{g}, a{a}
    {}
    operator float4()
    {
        return float4(r, g, b, a);
    }
    operator XMVECTOR()
    {
        return _mm_set_ps(a, b, g, r);
    }
    BGRA& operator=(const BGRA& bgra)
    {       
        b = bgra.b; g = bgra.g; r = bgra.r; a = bgra.a;
        return *this;
    }
    BGRA& operator=(const float4& c)
    {
        r = (uint8_t)c.x; g = (uint8_t)c.y; b = (uint8_t)c.z; a = (uint8_t)c.w;
        return *this;
    }
    BGRA& operator=(const FXMVECTOR& c)
    {
        r = (uint8_t)XMVectorGetX(c);
        g = (uint8_t)XMVectorGetY(c);
        b = (uint8_t)XMVectorGetZ(c);
        a = (uint8_t)XMVectorGetW(c);
        return *this;
    }
};

/// <summary>
/// Flying Camera
/// </summary>
class Camera
{
public:
    Camera()  
    {
      //default camera parameters
      SetLookAt(float3(30.0f, 15.0f, 30.01f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f)); // the camera is always looking at the origin (0,0,0)
      SetPerspective(45.0f, (float)ImageWidth / ImageHeight, nearClippingPLane, farClippingPLane);     
    }
    ~Camera() {}

    float  GetFov() { return FovY; }
    void SetPosition(float x, float y, float z) { position = float3(x, y, z); }

    float3 GetPosition() const { return position; };

    float4x4 GetViewMatrix() const { return View; };

    float4x4 GetProjectionMatrix() const { return Perspective; };

    float4x4 SetPerspective(float fov, float aspectRatio, float nearZ, float farZ, bool Right=true) 
    {
        
        FovY = fov;
        AspectRatio = aspectRatio;
        NearClipPlane = nearZ;
        FarClipPlane = farZ;
        if (Right) // if  we use right-handed coordinate system 
           Perspective = PerspectiveFovRH(fov, aspectRatio, nearZ, farZ);
        else
           Perspective = PerspectiveFovLH(fov, aspectRatio, nearZ, farZ);
        return Perspective;
    }

    float4x4 SetLookAt(float3 const& eyeposition, float3 const& target, float3 const& up, bool Right=true)
    {
        
        if (Right)
            View = MatrixLookAtRH(eyeposition, target, up);
        else
            View = MatrixLookAtLH(eyeposition, target, up);
        right_vec = float3(View.m11, View.m21, View.m31);
        up_vec = float3(View.m12, View.m22, View.m32);
        look_vec = float3(View.m13, View.m23, View.m33);
        position = eyeposition;
        return View;
    }

    void Strafe(float distance)
    {
        position += distance* right_vec ;
    }

    void Walk(float distance)
    {
        position += distance * look_vec ;
    }

    void RotateUpDown(float angle)
    {
        // rotate up-vector and look-vector about the right-vector
        float4x4 R = make_float4x4_from_axis_angle(right_vec, -angle);
        up_vec = transform_normal(up_vec, R);
        look_vec = transform_normal(look_vec, R);
    }

    void RotateLeftRight(float angle)
    {
        // rotate basis vectors about Y-axis of the world coordinate
        float4x4 R = make_float4x4_rotation_y(-angle);
     
        right_vec = transform_normal(right_vec, R);
        up_vec = transform_normal(up_vec, R);
        look_vec = transform_normal(look_vec, R);

    }

    void Update()
    {
        float3 R = right_vec;
        float3 U = up_vec;
        float3 L = look_vec;
        float3 P = position;

        L = normalize(L);
      
            U = normalize(cross(L, R));
            R = cross(U, L);
            right_vec = R; up_vec = U; look_vec = L;
            View.m11 = R.x;
            View.m21 = R.y;
            View.m31 = R.z;
            View.m41 = -dot(R, P);

            View.m12 = U.x;
            View.m22 = U.y;
            View.m32 = U.z;
            View.m42 = -dot(U, P);

            View.m13 = L.x;
            View.m23 = L.y;
            View.m33 = L.z;
            View.m43 = -dot(L, P);

            View.m14 = 0.0f;
            View.m24 = 0.0f;
            View.m34 = 0.0f;
            View.m44 = 1.0f;
        
    }
private:

    float4x4 MatrixLookAtRH(float3 const& eye, float3 const& center, float3 const& updirection);
    float4x4 MatrixLookAtLH(float3 const& eye, float3 const& center, float3 const& updirection);
    float4x4 PerspectiveFovRH(const float& Fov, const float& aspectRatio, const float& zNear, const float& zFar);
    float4x4 PerspectiveFovLH(const float& Fov, const float& aspectRatio, const float& zNear, const float& zFar);

    float3 position{}, right_vec{ 1.0f,0.0f,0.0f }, up_vec{0.0f,1.0f,0.0f}, look_vec{0.0f,0.0f,1.0f};
    float FovY{}, AspectRatio{}, NearClipPlane{}, FarClipPlane{};
    float4x4 View = float4x4::identity() , Perspective = float4x4::identity();
};

struct Material
{
    float4 emissive;
    float4 ambient;
    float4 diffuse;
    float4 specular;
   
    float power;
    bool useTexture;
    // define a gold material
    Material() : emissive{ 0.0f, 0.0f, 0.0f, 1.0f },
        ambient{ 0.24725f, 0.1995f, 0.0745f, 1.0f },
        diffuse{ 0.75164f, 0.60648f, 0.22648f, 1.0f },
        specular{ 0.628281f, 0.555802f, 0.366065f, 1.0f },
        power(21.2f),
        useTexture(false)
    {}
    Material(float4 e, float4 a, float4 d, float4 s, float pow, float useTex) : emissive{ e },
        ambient{ a }, diffuse{ d }, specular{ s }, power(pow), useTexture(useTex)
    {}
};

struct Light
{
    float3 position;
    float3 direction;
    float4 color;
    float constantAttenuation;
    float linearAttenuation;
    float quadraticAttenuation;
    float globalAmbient;
  
    Light() : position{ 0.0f, 0.0f, 0.0f },
        direction{ 0.0f, 0.0f, 0.0f },
        color{ 1.0f, 1.0f, 1.0f, 1.0f },
        constantAttenuation(1.0f),
        linearAttenuation(0.001f),
        quadraticAttenuation(0.0f),       
        globalAmbient(1.0f)
    {}
    Light(float3 pos, float3 dir, float4 col, float ga, float Kc, float Kl, float Kq) : 
        position{ pos },
        direction{ dir },
        color{ col },
        constantAttenuation(Kc),
        linearAttenuation(Kl),
        quadraticAttenuation(Kq),
        globalAmbient(ga)
    {}
};

struct VertexIndex
{
    uint32_t v0, v1, v2;
};

struct Vertex
{
    float4 position; // clip-space position
    float3 view_position; // view-space position
    float4 world_position; // wolrd-space position
    float4 normal; // world-space normal
    float2 texcoord;
};
struct Triangle
{
    Vertex v0, v1, v2;
    Triangle(Vertex& v_0, Vertex& v_1, Vertex& v_2) : v0(v_0), v1(v_1), v2(v_2) {}
};

 class Mesh
{
public:
    vector<float3> positions, view_positions; // local-space positions
    vector<float4> world_positions, raster_positions, clip_space_positions{};
    vector<float3> normals; // local-space normals
    vector<float4> world_normals{};
    vector<float2> textureCoords{};
    vector<uint32_t> indices{};
    vector<uint32_t> texcoordIndices{};
    uint32_t number_of_triangles{};
    Material mat;
    float4x4 World;
    float4x4 InverseTransposeWorld;
    Mesh() : positions{ vector<float3>(0) }, normals{ vector<float3>(0) }, textureCoords{ vector<float2>(0) },
        indices{ vector<uint32_t>(0) }, texcoordIndices{ vector<uint32_t>(0) }, number_of_triangles{ 0 }, mat{ Material() }, World{ float4x4() }, InverseTransposeWorld{ float4x4() }
    {     
    }
    Mesh(vector<float3> pos, vector<float3> norm, vector<float2>texcoord, vector<uint32_t> ind, vector<uint32_t> texInd, uint32_t numtriangles, Material material,
        float4x4 worldTransform, float4x4 InvTransWorld):
        positions{pos}, normals{ norm }, textureCoords{ texcoord }, indices{ ind }, texcoordIndices{ texInd }, number_of_triangles{ numtriangles }, mat{ material }, 
        World{ worldTransform }, InverseTransposeWorld{ InvTransWorld }
    {
    }
    Mesh(const Mesh& m) : positions{ m.positions }, normals{ m.normals }, textureCoords{ m.textureCoords }, 
        indices{ m.indices }, texcoordIndices{ m.texcoordIndices }, number_of_triangles{ m.number_of_triangles }, mat{ m.mat }, 
        World{ m.World }, InverseTransposeWorld {m.InverseTransposeWorld}
    {}
    Mesh& operator=(const Mesh& m)
    {
        positions = m.positions;
        normals = m.normals;
        textureCoords = m.textureCoords;
        indices = m.indices;
        texcoordIndices = m.texcoordIndices;
        number_of_triangles = m.number_of_triangles;
        mat = m.mat;
        World = m.World;
        InverseTransposeWorld = m.InverseTransposeWorld;
        return *this;
    }
   
    void CreateMeshFromArray(uint32_t numtriangles, float3* pos, size_t pos_size, float3* norm, size_t norm_size, float2* texcoord, size_t tex_size, 
            uint32_t* index , size_t index_size, uint32_t* texIndex, size_t texindex_size)
    {
        number_of_triangles = numtriangles;
        if (pos != nullptr && norm != nullptr && texcoord != nullptr && index != nullptr && texIndex!=nullptr)
        {
            for (size_t i = 0; i < pos_size; i++)
                positions.push_back(pos[i]);            
            for (size_t i = 0; i < norm_size; i++)            
                normals.push_back(norm[i]);
            for (size_t i = 0; i < tex_size; i++)
                textureCoords.push_back(texcoord[i]);
            for (size_t i = 0; i < index_size; i++)
                indices.push_back(index[i]);
            for (size_t i = 0; i < texindex_size; i++)
                texcoordIndices.push_back(texIndex[i]);
        }
    }
    void ComputeVertexNormals(float3* vertices, uint32_t* indices, uint32_t numvertices, uint32_t numtriangles, float3* vertexnormals);
    void CreateSphere(float radius, int slices, int stacks);
};


class Render
{  

private:
    float4x4 WorldToCamera = float4x4::identity(),
        CameraToScreen = float4x4::identity();
    float4x4 Identity = float4x4::identity();
    float t = 0, b = 0, l = 0, r = 0;   
    vector<BGRA> frameBuffer{};
    vector<float> depthBuffer{};
    vector<Mesh> meshes;
    float3 viewerPosition = float3(25.0f, 15.0f, 30.0f);
    
    Light light1;

    /*inline float min3(const float& a, const float& b, const float& c)
    {
        return min(a, min(b, c));
    }

    inline float max3(const float& a, const float& b, const float& c)
    {
        return max(a, max(b, c));
    }*/
    /// <summary>
    /// Edge equation calculation for right-handed coordinate system
    /// </summary>
    /// <param name="a">starting point of the edge</param>
    /// <param name="b">ending point  of the edge </param>
    /// <param name="c">Point coordinate within the triangle</param>
    /// for left-handed system: (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
    /// <returns></returns>
   
    /*inline float edgeFunc(const float3& a, const float3& b, const float3& p)
    {
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);// triangle edge equation: (px - X) * dY - (py - Y) * dX
    }*/

    // Matrices for right- and left-handed systems
    //float4x4 MatrixLookAtRH(float3 const& eye, float3 const& center, float3 const& updirection);
    //float4x4 MatrixLookAtLH(float3 const& eye, float3 const& center, float3 const& updirection);
    //float4x4 PerspectiveFovRH(const float& Fov, const float& AspectRatio, const float& zNear, const float& zFar);
    //float4x4 PerspectiveFovLH(const float& Fov, const float& AspectRatio, const float& zNear, const float& zFar);

    /*
    // compute screen-space coordinates without matrices using real camera parameters
    void computeScreenCoordinates(
        const float& filmApertureWidth,
        const float& filmApertureHeight,
        const uint32_t& imageWidth,
        const uint32_t& imageHeight,
        const FitResolutionGate& fitFilm,
        const float& nearClippingPLane,
        const float& focalLength,
        float& top, float& bottom, float& left, float& right
    );
    */
public:
    Render() : flyCamera()
    {
        Mesh sphere1, sphere2;

        // create mesh for globe and use default material (Gold)
        sphere1.CreateSphere(5.0f, numslices, numstacks);
        sphere1.World = make_float4x4_translation(float3(-10.0f, 0.0f, 0.0f));
        // copper material
        // sphere.mat = Material(float4(0.0f,0.0f,0.0f,1.0f), float4(0.2125f, 0.1275f, 0.054f, 1.0f), float4(0.714f, 0.4284f, 0.18144f,1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f),55.8f, false);


        sphere2.CreateSphere(5.0f, numslices, numstacks);
        sphere2.World = make_float4x4_translation(float3(10.0f, 0.0f, 0.0f));
        sphere2.mat = Material(float4(0.0f, 0.0f, 0.0f, 1.0f), float4(0.2125f, 0.1275f, 0.054f, 1.0f), float4(0.714f, 0.4284f, 0.18144f, 1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f), 55.8f, false);
        //  sphere2.mat = Material(float4(0.0f, 0.0f, 0.0f, 1.0f), float4(0.2125f, 0.1275f, 0.054f, 1.0f), float4(0.714f, 0.4284f, 0.18144f, 1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f), 55.8f, false);
        //sphere2.mat = Material(float4(0.0f,0.0f,0.0f,1.0f), float4(0.25f, 0.20725f, 0.20725f, 1.0f), float4(1.0f, 0.829f, 0.829f, 1.0f), float4(0.296648f, 0.296648f, 0.296648f, 1.0f), 11.264f, false);

        meshes.push_back(sphere1);
        meshes.push_back(sphere2);
    }
    void RenderMain(double* time_passed);
    void TransformVertices(Mesh* mesh);
    /*void projection(
        const float3& vertexWorld,
        const float4x4& worldToCamera,
        const float& l,
        const float& r,
        const float& t,
        const float& b,
        const float& nearplane,
        const uint32_t& imageWidth,
        const uint32_t& imageHeight,
        float3& vertexRaster,
        float3& vertexCamera
    );
    void Projection(
        const XMVECTOR& vertexWorld,
        const uint32_t& imageWidth,
        const uint32_t& imageHeight,
        XMFLOAT3& vertexRaster,
        XMFLOAT3& vertexCamera
    );*/

    void Project(const XMVECTOR& vertexWorld, XMFLOAT3& vertexCamera, XMFLOAT4& vertexNDC);
    void PerspectiveDivide(const XMFLOAT3& vertexCamera, XMVECTOR& vertexNDC, XMFLOAT3& vertexRaster, const uint32_t& imageWidth,
        const uint32_t& imageHeight);
    void DrawMesh(Mesh* mesh);
  
    void SetViewPosition(float3 pos) {  viewerPosition = pos; }
    float3 GetViewPosition() { return viewerPosition; }
    Camera flyCamera;
    BGRA* p_bgra = nullptr;
};



