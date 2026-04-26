#pragma once
#include "pch.h"

using namespace std;
using namespace winrt;
using namespace Windows;
using namespace Windows::Foundation::Numerics;
using namespace DirectX;
using namespace Concurrency;

constexpr int numslices = 50;
constexpr int numstacks = 50;
extern uint32_t ImageWidth;
extern uint32_t ImageHeight;
constexpr float nearClippingPLane = 0.1f;
constexpr float farClippingPLane = 1000.f;

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
      SetLookAt(float3(30.1f, 15.0f, 30.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f)); // the camera is always looking at the origin (0,0,0)
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
        right_vec = float3(View.m11, View.m21, View.m31); // u
        up_vec = float3(View.m12, View.m22, View.m32); // v
        look_vec = float3(View.m13, View.m23, View.m33); // w
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

    float3 position{}, 
        right_vec{ 1.0f,0.0f,0.0f }, /*u basis vector*/
        up_vec{0.0f,1.0f,0.0f}, /*v basis vector*/
        look_vec{0.0f,0.0f,1.0f}; /*w basis vector*/
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
    float4 world_position; // world-space position
    float4 normal; // world-space normal
    float2 texcoord;
    float3 tangent;
    float3 bitangent;
};

struct VertexPacket8
{
    __m256 x, y, z;      // position
    __m256 nx, ny, nz;   // normal
    __m256 u, v;         // texcoord;
};

//it needs for clipTriangle() only
struct Triangle
{
   alignas(16) Vertex v0, v1, v2;
    Triangle(Vertex& v_0, Vertex& v_1, Vertex& v_2) : v0(v_0), v1(v_1), v2(v_2) {}
};

//
//* Csak minta!!!!!!!
struct Mat4
{
    // row-major: m[row*4 + col]
    float m[16];
};





 class Mesh
{
public:
    //----------------------------------------------------------
    // New Array-of-Structures vertex storage for graphics friendly layout
    struct MeshVertexAoS
    {
        float3 position;       // local-space position
        float3 normal;         // local-space normal
        float2 texcoord;       // texture coordinate
        float3 tangent;         // object local tangent
        float3 bitangent;       // object local bitangent
        float4 world_position; // world-space position (filled in TransformVertices)
        float4 world_normal;   // world-space normal (filled in TransformVertices)
        float3 view_position;  // view-space position (filled in TransformVertices)
        float4 clip_position;  // clip-space position (filled in TransformVertices)
    };
    alignas(16) vector<MeshVertexAoS> vertices; // per-vertex attributes (AoS)
    alignas(16) vector<uint32_t> indices{}; // triangle indices
   //-----------------------------------------------------------

    //----------------------------------------------------------
    // vertex structure SoA (Structure-of-Arrays) for AVX
    struct MeshSoA
    {
        // N darab vertex
        std::vector<float> posX; //positions
        std::vector<float> posY;
        std::vector<float> posZ;

        std::vector<float> nrmX; // normals
        std::vector<float> nrmY;
        std::vector<float> nrmZ;

        std::vector<float> uvU; // texture-coords
        std::vector<float> uvV;

        // 3-asával indexek (triangle list)
        std::vector<uint32_t> indices;

        uint32_t VertexCount() const { return (uint32_t)posX.size(); }
        uint32_t IndexCount()  const { return (uint32_t)indices.size(); }
    };
    alignas(16) vector<MeshSoA*> SoA_meshes;// per-vertex attributes (SoA)

    //----------------------------------------------------------


    uint32_t number_of_triangles{};
    Material mat;
   alignas(16) float4x4 World;
   alignas(16) float4x4 InverseTransposeWorld;
    Mesh() : vertices{ vector<MeshVertexAoS>(0) }, indices{ vector<uint32_t>(0) }, number_of_triangles{ 0 }, mat{ Material() }, World{ float4x4() }, InverseTransposeWorld{ float4x4() }
    {     
    }
    Mesh(vector<MeshVertexAoS> verts, vector<uint32_t> ind, uint32_t numtriangles, Material material,
        float4x4 worldTransform, float4x4 InvTransWorld):
        vertices{verts}, indices{ ind }, number_of_triangles{ numtriangles }, mat{ material }, 
        World{ worldTransform }, InverseTransposeWorld{ InvTransWorld }
    {
    }
    Mesh(const Mesh& m) : vertices{ m.vertices },
        indices{ m.indices }, number_of_triangles{ m.number_of_triangles }, mat{ m.mat }, 
        World{ m.World }, InverseTransposeWorld {m.InverseTransposeWorld}
    {}
    Mesh& operator=(const Mesh& m)
    {
        // copy new AOS vertices if present, otherwise keep empty
        vertices = m.vertices;
        indices = m.indices;
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
            // build AOS vertices from provided arrays
            size_t count = min(pos_size, min(norm_size, tex_size));
            vertices.reserve(count);
            for (size_t i = 0; i < count; i++)
            {
                MeshVertexAoS v{};
                v.position = pos[i];
                v.normal = norm[i];
                v.texcoord = texcoord[i];
                v.tangent = float3(0.0f);
                v.bitangent = float3(0.0f);
                vertices.push_back(v);
            }
            for (size_t i = 0; i < index_size; i++)
                indices.push_back(index[i]);
        }
    }
    void ComputeVertexNormals();
    void CreateSphere(float radius, int slices, int stacks);
    void CreateSphere2(float radius, int slices, int stacks);
    MeshSoA CreateSphereSoA(uint32_t stacks, uint32_t slices, float radius);
};

 struct Tile {
     uint32_t minX, minY;
     uint32_t maxX, maxY;
 };
 // mesh data after transformation
 struct MeshSoAFrame
 {
     // clip-space
     std::vector<float> clipX;
     std::vector<float> clipY;
     std::vector<float> clipZ;
     std::vector<float> clipW;

     // view-space
     std::vector<float> viewX;
     std::vector<float> viewY;
     std::vector<float> viewZ;

     // world-space
     std::vector<float> worldX;
     std::vector<float> worldY;
     std::vector<float> worldZ;

     // world-space normal
     std::vector<float> nrmX;
     std::vector<float> nrmY;
     std::vector<float> nrmZ;

     // UV
     std::vector<float> uvU;
     std::vector<float> uvV;

     // index buffer változatlanul
     std::vector<uint32_t> indices;

     uint32_t VertexCount() const { return (uint32_t)clipX.size(); }
     uint32_t IndexCount()  const { return (uint32_t)indices.size(); }
 };
 struct TriWork
 {
     XMVECTOR v0, v1, v2;
     XMVECTOR n0, n1, n2;
     XMVECTOR tex0, tex1, tex2;
     XMVECTOR v0Raster, v1Raster, v2Raster;
     XMVECTOR v0RasterZ, v1RasterZ, v2RasterZ;
     uint32_t bb_xmin, bb_xmax, bb_ymin, bb_ymax;
 };
class Render
{  

private:
    alignas(16) float4x4 WorldToCamera = float4x4::identity(),
        CameraToScreen = float4x4::identity();
    alignas(16) float4x4 Identity = float4x4::identity();
    float t = 0, b = 0, l = 0, r = 0;   
    vector<BGRA> frameBuffer{};
    vector<float> depthBuffer{};
    vector<Mesh> meshes;
    float3 viewerPosition = float3(25.0f, 15.0f, 30.0f);
    
    Light light1;
   
    /// <summary>
    /// Edge equation calculation for right-handed coordinate system
    /// </summary>
    /// <param name="a">starting point of the edge</param>
    /// <param name="b">ending point  of the edge </param>
    /// <param name="p">Point coordinate within the triangle</param>
    /// for left-handed system: (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
    /// <returns></returns>
   
    /*inline float edgeFunc(const float3& a, const float3& b, const float3& p)
    {
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);// triangle edge equation: (px - X) * dY - (py - Y) * dX
    }*/
   

public:
    Render() : flyCamera()
    {
        Mesh sphere1, sphere2, sphere3, sphere4;

        // create mesh for globe and use default material (Gold)
        sphere1.CreateSphere(5.0f, numslices, numstacks);
        sphere1.World = make_float4x4_translation(float3(-10.0f, 0.0f, 0.0f));
        XMStoreFloat4x4(&sphere1.InverseTransposeWorld, XMMatrixTranspose(XMMatrixInverse(nullptr, XMLoadFloat4x4(&sphere1.World))));

        // copper material
        // sphere.mat = Material(float4(0.0f,0.0f,0.0f,1.0f), float4(0.2125f, 0.1275f, 0.054f, 1.0f), float4(0.714f, 0.4284f, 0.18144f,1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f),55.8f, false);


        sphere2.CreateSphere(5.0f, numslices, numstacks);
        sphere2.World = make_float4x4_translation(float3(10.0f, 0.0f, 0.0f));
        XMStoreFloat4x4(&sphere2.InverseTransposeWorld, XMMatrixTranspose(XMMatrixInverse(nullptr, XMLoadFloat4x4(&sphere2.World))));
        sphere2.mat = Material(float4(0.0f, 0.0f, 0.0f, 1.0f), float4(0.2125f, 0.1275f, 0.054f, 1.0f), float4(0.714f, 0.4284f, 0.18144f, 1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f), 55.8f, false);
        //  sphere2.mat = Material(float4(0.0f, 0.0f, 0.0f, 1.0f), float4(0.2125f, 0.1275f, 0.054f, 1.0f), float4(0.714f, 0.4284f, 0.18144f, 1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f), 55.8f, false);
        //sphere2.mat = Material(float4(0.0f,0.0f,0.0f,1.0f), float4(0.25f, 0.20725f, 0.20725f, 1.0f), float4(1.0f, 0.829f, 0.829f, 1.0f), float4(0.296648f, 0.296648f, 0.296648f, 1.0f), 11.264f, false);

        sphere3.CreateSphere(5.0f, numslices, numstacks);
        sphere3.World = make_float4x4_translation(float3(10.0f, 0.0f, 20.0f));
        XMStoreFloat4x4(&sphere3.InverseTransposeWorld, XMMatrixTranspose(XMMatrixInverse(nullptr, XMLoadFloat4x4(&sphere3.World))));
        sphere3.mat = Material(float4(0.0f, 0.0f, 0.0f, 1.0f), float4(0.125f, 0.275f, 0.54f, 1.0f), float4(0.714f, 0.4284f, 0.18144f, 1.0f), float4(0.393548f, 0.271906f, 0.166721f, 1.0f), 55.8f, false);
        
        sphere4.CreateSphere(5.0f, numslices, numstacks);
        sphere4.World = make_float4x4_translation(float3(-10.0f, 0.0f, 20.0f));
        XMStoreFloat4x4(&sphere4.InverseTransposeWorld, XMMatrixTranspose(XMMatrixInverse(nullptr, XMLoadFloat4x4(&sphere4.World))));
        sphere4.mat = Material(float4(0.0f, 0.0f, 0.0f, 1.0f), float4(0.25f, 0.20725f, 0.20725f, 1.0f), float4(1.0f, 0.829f, 0.829f, 1.0f), float4(0.296648f, 0.296648f, 0.296648f, 1.0f), 11.264f, false);

        meshes.push_back(sphere1);
        meshes.push_back(sphere2);
        meshes.push_back(sphere3);
        meshes.push_back(sphere4);
        
    }
    //void BuildTiles(uint32_t tileSize);
    void RenderMain(double* time_passed);
    void TransformVertices(Mesh* mesh);
    void Project(const XMVECTOR& vertexWorld, XMFLOAT3& vertexCamera, XMFLOAT4& vertexNDC);
    void PerspectiveDivide(const XMFLOAT3& vertexCamera, XMVECTOR& vertexNDC, XMFLOAT3& vertexRaster, const uint32_t& imageWidth,
        const uint32_t& imageHeight);

    void DrawMesh(Mesh* mesh);
    // Tile-based parallel renderer: spawns as many tiles/threads as logical processors
    void DrawMeshTiled(Mesh* mesh);
    void DrawMeshTiledBinning(Mesh* mesh);
    void DrawMeshTiledBinningSoA(const MeshSoAFrame& mesh, const Material& mat);
    void SetViewPosition(float3 pos) {  viewerPosition = pos; }
    float3 GetViewPosition() { return viewerPosition; }
    Camera flyCamera;
    BGRA* p_bgra = nullptr;
    std::vector<Tile> tiles;

    //AOS → SoA transzformáció
    //Ezt hívod meg közvetlenül a vertex‑transzformáció után, per mesh, per frame.
    MeshSoAFrame BuildSoAFrameFromMesh(const Mesh& mesh)
    {
        MeshSoAFrame soa;

        const uint32_t n = (uint32_t)mesh.vertices.size();

        soa.clipX.resize(n);
        soa.clipY.resize(n);
        soa.clipZ.resize(n);
        soa.clipW.resize(n);

        soa.viewX.resize(n);
        soa.viewY.resize(n);
        soa.viewZ.resize(n);

        soa.worldX.resize(n);
        soa.worldY.resize(n);
        soa.worldZ.resize(n);

        soa.nrmX.resize(n);
        soa.nrmY.resize(n);
        soa.nrmZ.resize(n);

        soa.uvU.resize(n);
        soa.uvV.resize(n);

        soa.indices = mesh.indices; // sima másolat

        for (uint32_t i = 0; i < n; ++i)
        {
            const auto& v = mesh.vertices[i];

            soa.clipX[i] = v.clip_position.x;
            soa.clipY[i] = v.clip_position.y;
            soa.clipZ[i] = v.clip_position.z;
            soa.clipW[i] = v.clip_position.w;

            soa.viewX[i] = v.view_position.x;
            soa.viewY[i] = v.view_position.y;
            soa.viewZ[i] = v.view_position.z;

            soa.worldX[i] = v.world_position.x;
            soa.worldY[i] = v.world_position.y;
            soa.worldZ[i] = v.world_position.z;

            soa.nrmX[i] = v.world_normal.x;
            soa.nrmY[i] = v.world_normal.y;
            soa.nrmZ[i] = v.world_normal.z;

            soa.uvU[i] = v.texcoord.x;
            soa.uvV[i] = v.texcoord.y;
        }

        return soa;
    }
    //Segédfüggvény: tile‑sarkok + edge‑teszt
    //Ez konzervatív: ha true, akkor a tile biztosan teljesen kívül van; ha false, akkor lehet, hogy csak egy pixel ér bele, de azt már a pixelloop eldönti.
    inline bool TileOutsideTriangle(
        const TriWork& tw,
        uint32_t tileX0, uint32_t tileY0,
        uint32_t tileX1, uint32_t tileY1)
    {
        XMFLOAT4 v0r, v1r, v2r;
        XMStoreFloat4(&v0r, tw.v0Raster);
        XMStoreFloat4(&v1r, tw.v1Raster);
        XMStoreFloat4(&v2r, tw.v2Raster);

        float dX01 = v1r.x - v0r.x;
        float dY01 = v1r.y - v0r.y;
        float dX12 = v2r.x - v1r.x;
        float dY12 = v2r.y - v1r.y;
        float dX20 = v0r.x - v2r.x;
        float dY20 = v0r.y - v2r.y;

        // area ugyanúgy, mint a rasterizálásnál
        float area = (v2r.x - v0r.x) * (v1r.y - v0r.y) - (v2r.y - v0r.y) * (v1r.x - v0r.x);
        if (area <= 0.0f)
        {
            // back-face vagy degenerált → NEM cullolunk tile-t
            return false;
        }

        float cx[4] = {
            tileX0 + 0.5f, tileX1 + 0.5f,
            tileX0 + 0.5f, tileX1 + 0.5f
        };
        float cy[4] = {
            tileY0 + 0.5f, tileY0 + 0.5f,
            tileY1 + 0.5f, tileY1 + 0.5f
        };

        auto edgeTest = [](float px, float py,
            float vx, float vy,
            float dX, float dY)
            {
                return (px - vx) * dY - (py - vy) * dX;
            };

        // edge v1→v2, teszt v0
        int out0 = 0;
        for (int i = 0; i < 4; ++i)
            out0 += (edgeTest(cx[i], cy[i], v1r.x, v1r.y, dX12, dY12) < 0.0f);
        if (out0 == 4)
            return true;

        // edge v2→v0, teszt v1
        int out1 = 0;
        for (int i = 0; i < 4; ++i)
            out1 += (edgeTest(cx[i], cy[i], v2r.x, v2r.y, dX20, dY20) < 0.0f);
        if (out1 == 4)
            return true;

        // edge v0→v1, teszt v2
        int out2 = 0;
        for (int i = 0; i < 4; ++i)
            out2 += (edgeTest(cx[i], cy[i], v0r.x, v0r.y, dX01, dY01) < 0.0f);
        if (out2 == 4)
            return true;

        return false;
    }

    // --- Packet8 betöltés SoA-ból ---

    inline VertexPacket8 LoadVertexPacket8(const Mesh::MeshSoA& mesh, uint32_t baseIndex)
    {
        VertexPacket8 p{};

        float px[8], py[8], pz[8];
        float nx[8], ny[8], nz[8];
        float uu[8], vv[8];

        for (int i = 0; i < 8; ++i)
        {
            uint32_t idx = baseIndex + i;
            px[i] = mesh.posX[idx];
            py[i] = mesh.posY[idx];
            pz[i] = mesh.posZ[idx];

            nx[i] = mesh.nrmX[idx];
            ny[i] = mesh.nrmY[idx];
            nz[i] = mesh.nrmZ[idx];

            uu[i] = mesh.uvU[idx];
            vv[i] = mesh.uvV[idx];
        }

        p.x = _mm256_loadu_ps(px);
        p.y = _mm256_loadu_ps(py);
        p.z = _mm256_loadu_ps(pz);
        p.nx = _mm256_loadu_ps(nx);
        p.ny = _mm256_loadu_ps(ny);
        p.nz = _mm256_loadu_ps(nz);
        p.u = _mm256_loadu_ps(uu);
        p.v = _mm256_loadu_ps(vv);

        return p;
    }

    // --- WVP transzformáció Packet8-ra --- World-View-Projection

    inline void TransformPacket8_WVP(
        const VertexPacket8& in,
        const float4x4& wvp,
        __m256& clipX,
        __m256& clipY,
        __m256& clipZ,
        __m256& clipW)
    {
        __m256 x = in.x;
        __m256 y = in.y;
        __m256 z = in.z;

        __m256 m00 = _mm256_set1_ps(wvp.m11);
        __m256 m01 = _mm256_set1_ps(wvp.m12);
        __m256 m02 = _mm256_set1_ps(wvp.m13);
        __m256 m03 = _mm256_set1_ps(wvp.m14);

        __m256 m10 = _mm256_set1_ps(wvp.m21);
        __m256 m11 = _mm256_set1_ps(wvp.m22);
        __m256 m12 = _mm256_set1_ps(wvp.m23);
        __m256 m13 = _mm256_set1_ps(wvp.m24);

        __m256 m20 = _mm256_set1_ps(wvp.m31);
        __m256 m21 = _mm256_set1_ps(wvp.m32);
        __m256 m22 = _mm256_set1_ps(wvp.m33);
        __m256 m23 = _mm256_set1_ps(wvp.m34);

        __m256 m30 = _mm256_set1_ps(wvp.m41);
        __m256 m31 = _mm256_set1_ps(wvp.m42);
        __m256 m32 = _mm256_set1_ps(wvp.m43);
        __m256 m33 = _mm256_set1_ps(wvp.m44);

        // clipX = m00*x + m01*y + m02*z + m03  -> for column-major matrix
       // clipX = x*m00 + y*m01 + z*m02 + m03  -> for row-major matrix (like float4x4)

        clipX = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m00),
                _mm256_mul_ps(y, m01)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m02),
                m03));

        // clipY = m10*x + m11*y + m12*z + m13
        clipY = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m10),
                _mm256_mul_ps(y, m11)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m12),
                m13));

        // clipZ = m20*x + m21*y + m22*z + m23
        clipZ = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m20),
                _mm256_mul_ps(y, m21)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m22),
                m23));

        // clipW = m30*x + m31*y + m32*z + m33
        clipW = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m30),
                _mm256_mul_ps(y, m31)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m32),
                m33));
    }


    inline void TransformPacket8_World(
        const VertexPacket8& in,
        const float4x4& world,
        __m256& wrldX,
        __m256& wrldY,
        __m256& wrldZ
        )
    {
        __m256 x = in.x;
        __m256 y = in.y;
        __m256 z = in.z;

        __m256 m00 = _mm256_set1_ps(world.m11);
        __m256 m01 = _mm256_set1_ps(world.m12);
        __m256 m02 = _mm256_set1_ps(world.m13);
        __m256 m03 = _mm256_set1_ps(world.m14);

        __m256 m10 = _mm256_set1_ps(world.m21);
        __m256 m11 = _mm256_set1_ps(world.m22);
        __m256 m12 = _mm256_set1_ps(world.m23);
        __m256 m13 = _mm256_set1_ps(world.m24);

        __m256 m20 = _mm256_set1_ps(world.m31);
        __m256 m21 = _mm256_set1_ps(world.m32);
        __m256 m22 = _mm256_set1_ps(world.m33);
        __m256 m23 = _mm256_set1_ps(world.m34);

        __m256 m30 = _mm256_set1_ps(world.m41);
        __m256 m31 = _mm256_set1_ps(world.m42);
        __m256 m32 = _mm256_set1_ps(world.m43);
        __m256 m33 = _mm256_set1_ps(world.m44);

        // worldX = m00*x + m01*y + m02*z + m03  -> for column-major matrix
       // worldX = x*m00 + y*m01 + z*m02 + m03  -> for row-major matrix (like float4x4)

        wrldX = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m00),
                _mm256_mul_ps(y, m01)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m02),
                m03));

        // worldY = m10*x + m11*y + m12*z + m13
        wrldY = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m10),
                _mm256_mul_ps(y, m11)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m12),
                m13));

        // worldZ = m20*x + m21*y + m22*z + m23
        wrldZ = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x, m20),
                _mm256_mul_ps(y, m21)),
            _mm256_add_ps(
                _mm256_mul_ps(z, m22),
                m23));
     
    }
    // transform normal vectors with Inverse of Transpose of World Matrix
    inline void TransformPacket8_Normal(
        const VertexPacket8& in,
        const float4x4& invTransWorld,
        __m256& normX,
        __m256& normY,
        __m256& normZ)
    {
        __m256 x = in.x;
        __m256 y = in.y;
        __m256 z = in.z;

        __m256 m00 = _mm256_set1_ps(invTransWorld.m11);
        __m256 m01 = _mm256_set1_ps(invTransWorld.m12);
        __m256 m02 = _mm256_set1_ps(invTransWorld.m13);

        __m256 m10 = _mm256_set1_ps(invTransWorld.m21);
        __m256 m11 = _mm256_set1_ps(invTransWorld.m22);
        __m256 m12 = _mm256_set1_ps(invTransWorld.m23);

        __m256 m20 = _mm256_set1_ps(invTransWorld.m31);
        __m256 m21 = _mm256_set1_ps(invTransWorld.m32);
        __m256 m22 = _mm256_set1_ps(invTransWorld.m33);


        // normX = m00*x + m01*y + m02*z   -> for column-major matrix
       // normX = x*m00 + y*m01 + z*m02   -> for row-major matrix (like float4x4)

        normX = _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(x, m00),
                                _mm256_mul_ps(y, m01)),            
                            _mm256_mul_ps(z, m02)
                );

        // normY = m10*x + m11*y + m12*z 
        normY = _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(x, m10),
                                _mm256_mul_ps(y, m11)),
                            _mm256_mul_ps(z, m12)
                );

        // normZ = m20*x + m21*y + m22*z
        normZ = _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(x, m20),
                                _mm256_mul_ps(y, m21)),
                            _mm256_mul_ps(z, m22)
                );      
    }

    // --- Clip → NDC → viewport Packet8 ---

    inline void ClipToScreenPacket8(
        __m256 clipX, __m256 clipY, __m256 clipZ, __m256 clipW,
        uint32_t imageW, uint32_t imageH,
        __m256& sx, __m256& sy, __m256& sz)
    {
        __m256 invW = _mm256_div_ps(_mm256_set1_ps(1.0f), clipW);

        __m256 ndcX = _mm256_mul_ps(clipX, invW);
        __m256 ndcY = _mm256_mul_ps(clipY, invW);
        __m256 ndcZ = _mm256_mul_ps(clipZ, invW);

        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 w = _mm256_set1_ps((float)imageW);
        __m256 h = _mm256_set1_ps((float)imageH);

        // sx = (ndcX * 0.5 + 0.5) * W
        __m256 tmpX = _mm256_add_ps(_mm256_mul_ps(ndcX, half), half);
        sx = _mm256_mul_ps(tmpX, w);

        // sy = (1 - (ndcY * 0.5 + 0.5)) * H
        __m256 tmpY = _mm256_add_ps(_mm256_mul_ps(ndcY, half), half);
        __m256 oneMinus = _mm256_sub_ps(one, tmpY);
        sy = _mm256_mul_ps(oneMinus, h);

        sz = ndcZ;
    }

    // --- Példa: teljes Packet8 vertex-pass egy mesh-en ---

    void TransformMeshPacket8_WVP_Normal_Viewport(
        Mesh::MeshSoA& mesh,
        const float4x4& wvp,
        const float4x4& world,
        const float4x4& invTransWorld,
        uint32_t imageW,
        uint32_t imageH,
        float* outScrX,
        float* outScrY,
        float* outScrZ,
        float* outNormX,
        float* outNormY,
        float* outNormZ,
        float* outWorldX,
        float* outWorldY,
        float* outWorldZ)
    {
        uint32_t n = mesh.VertexCount();
        uint32_t i = 0;

        for (; i + 7 < n; i += 8)
        {
            VertexPacket8 vp = LoadVertexPacket8(mesh, i);

            __m256 cx, cy, cz, cw;
            TransformPacket8_WVP(vp, wvp, cx, cy, cz, cw);
            __m256 wx, wy, wz;
            TransformPacket8_World(vp, world, wx, wy, wz); // compute world-position vector for lighting
            _mm256_storeu_ps(&outWorldX[i], wx);
            _mm256_storeu_ps(&outWorldY[i], wy);
            _mm256_storeu_ps(&outWorldZ[i], wz);
            __m256 nx, ny, nz;
            TransformPacket8_Normal(vp, invTransWorld, nx, ny, nz); // compute the right normal vector with inverse-transpose-normal
            _mm256_storeu_ps(&outNormX[i], nx);
            _mm256_storeu_ps(&outNormY[i], ny);
            _mm256_storeu_ps(&outNormZ[i], nz);

            __m256 sx, sy, sz;
            ClipToScreenPacket8(cx, cy, cz, cw, imageW, imageH, sx, sy, sz);

            _mm256_storeu_ps(&outScrX[i], sx);
            _mm256_storeu_ps(&outScrY[i], sy);
            _mm256_storeu_ps(&outScrZ[i], sz);
        }

        // maradék scalar
        for (; i < n; ++i)
        {
            float x = mesh.posX[i];
            float y = mesh.posY[i];
            float z = mesh.posZ[i];

            float nx = mesh.nrmX[i];
            float ny = mesh.nrmY[i];
            float nz = mesh.nrmZ[i];

            float X = x * wvp.m11 + y * wvp.m12 + z * wvp.m13  + wvp.m14;
            float Y = x * wvp.m21 + y * wvp.m22  + z * wvp.m23  + wvp.m24;
            float Z = x * wvp.m31 + y * wvp.m32 + z * wvp.m33 + wvp.m34;
            float W = x * wvp.m41 + y * wvp.m42  + z * wvp.m43  + wvp.m44;

            float NX = nx * invTransWorld.m11 + ny * invTransWorld.m12 + nz * invTransWorld.m13;
            float NY = nx * invTransWorld.m21 + ny * invTransWorld.m22 + nz * invTransWorld.m23;
            float NZ = nx * invTransWorld.m31 + ny * invTransWorld.m32 + nz * invTransWorld.m33;
            outNormX[i] = NX;
            outNormY[i] = NY;
            outNormZ[i] = NZ;

            float WX = x * world.m11 + y * world.m12 + z * world.m13 + world.m14;
            float WY = x * world.m21 + y * world.m22 + z * world.m23 + world.m24;
            float WZ = x * world.m31 + y * world.m32 + z * world.m33 + world.m34;
            outWorldX[i] = WX;
            outWorldY[i] = WY;
            outWorldZ[i] = WZ;

            float invW = 1.0f / W;
            float ndcX = X * invW;
            float ndcY = Y * invW;
            float ndcZ = Z * invW;

            float sx = (ndcX * 0.5f + 0.5f) * (float)imageW;
            float sy = (1.0f - (ndcY * 0.5f + 0.5f)) * (float)imageH;

            outScrX[i] = sx;
            outScrY[i] = sy;
            outScrZ[i] = ndcZ;

        }
    }
};



