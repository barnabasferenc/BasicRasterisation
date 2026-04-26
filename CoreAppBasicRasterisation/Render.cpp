#include "pch.h"
#include "Render.h"
#include <thread>



void Render::RenderMain(double* time_passed)
{    
 
    viewerPosition = flyCamera.GetPosition();
    WorldToCamera = flyCamera.GetViewMatrix();
    CameraToScreen = flyCamera.GetProjectionMatrix();

    light1.position = float3(30.0f, 0.0f, 0.0f);
    light1.direction = { 0.0f,0.0f,0.0f };
    light1.color = { 1.0f,1.0f,1.0f,1.0f }; //white colored light-source
    light1.globalAmbient = 1.0f;
    light1.constantAttenuation = 1.0f;
    light1.linearAttenuation = 0.01f;
    light1.quadraticAttenuation = 0.0f;
        
   // resize framebuffer and depth-buffer
    depthBuffer.assign((size_t)ImageWidth * ImageHeight, farClippingPLane);
    frameBuffer.assign((size_t)ImageWidth * ImageHeight, { 0,128,255,0 });

    MeshSoAFrame SoAFrame{};
    Material obj_mat{};
    auto t_start = std::chrono::high_resolution_clock::now(); 
   
        for (Mesh mesh : meshes)
        {
            TransformVertices(&mesh);
            SoAFrame = BuildSoAFrameFromMesh(mesh);
            obj_mat = mesh.mat;
            DrawMeshTiledBinningSoA(SoAFrame, obj_mat);
            //DrawMesh(&mesh);
            //DrawMeshTiledBinning(&mesh);
            //DrawMeshTiled(&mesh);
        }
   

    auto t_end = std::chrono::high_resolution_clock::now();
    auto passedTime = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    *time_passed = passedTime;
    //p_bgra is a public pointer to rendered image
    p_bgra = frameBuffer.data();
}


void Render::TransformVertices(Mesh* mesh)
{
    XMMATRIX World = XMLoadFloat4x4(&mesh->World);
    XMMATRIX InverseTransposeWorld = XMLoadFloat4x4(&mesh->InverseTransposeWorld);
   
    // Transform per-vertex attributes from local to world/view/clip space
    for (auto& mv : mesh->vertices)
    {
        // world normal
        float3 invn{};
        XMStoreFloat3(&invn, XMVector3TransformNormal(XMLoadFloat3(&mv.normal), InverseTransposeWorld));
        mv.world_normal = float4(invn, 0.0f);

        // world position
        XMVECTOR wv = XMVector3Transform(XMLoadFloat3(&mv.position), World);
        float3 wp{};
        XMStoreFloat3(&wp, wv);
        mv.world_position = float4(wp, 1.0f);

        XMFLOAT3 camera_p; XMFLOAT4 ndc_p;
        Project(wv, camera_p, ndc_p);
        mv.clip_position = float4(ndc_p.x, ndc_p.y, ndc_p.z, ndc_p.w);
        mv.view_position = float3(camera_p.x, camera_p.y, camera_p.z);
    }
 
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> 
//The Sutherland - Hodgman algorithm is traditionally used for 2D polygon clipping, but it can be extended to 3D for clipping triangles against a 3D clipping volume(e.g., a frustum).
// use homogeneous coordinates for positions
// Function to check if a point is inside the clipping plane
static bool isInside(const Vertex & point, const std::array<float, 4>&plane)
{
    return plane[0] * point.position.x + plane[1] * point.position.y + plane[2] * point.position.z + plane[3] >= 0;  // Ax * By * Cz + D >=0
}

// Function to compute the intersection of a line segment with a clipping plane
static Vertex intersect(const Vertex & p1, const Vertex & p2, const std::array<float, 4>&plane)
{
    Vertex v{};
    
    float d1 = plane[0] * p1.position.x + plane[1] * p1.position.y + plane[2] * p1.position.z + plane[3];  // dot3(p1.position, plane)
    float d2 = plane[0] * p2.position.x + plane[1] * p2.position.y + plane[2] * p2.position.z + plane[3]; // dot3(p2.position, plane)
    float t = d1 / (d1 - d2);

    //intersect all triangle`s attributes
    //float4(p1.position.x + t * (p2.position.x - p1.position.x),  p1.position.y + t * (p2.position.y - p1.position.y), p1.position.z + t * (p2.position.z - p1.position.z), p1.position.w + t * (p2.position.w - p1.position.w));
    v.position = lerp(p1.position, p2.position, t);
    //float3(p1.view_position.x + t * (p2.view_position.x - p1.view_position.x), p1.view_position.y + t * (p2.view_position.y - p1.view_position.y), p1.view_position.z + t * (p2.view_position.z - p1.view_position.z));   
    v.view_position = lerp(p1.view_position, p2.view_position, t);
    // float4(p1.world_position.x + t * (p2.world_position.x - p1.world_position.x),
    //                                    p1.world_position.y + t * (p2.world_position.y - p1.world_position.y), 
    //                                    p1.world_position.z + t * (p2.world_position.z - p1.world_position.z), 
    //                                    p1.world_position.w + t * (p2.world_position.w - p1.world_position.w));
    v.world_position = lerp(p1.world_position, p2.world_position, t);
    //float4(p1.normal.x + t * (p2.normal.x - p1.normal.x), p1.normal.y + t * (p2.normal.y - p1.normal.y), p1.normal.z + t * (p2.normal.z - p1.normal.z), p1.normal.w + t * (p2.normal.w - p1.normal.w));   
    v.normal = lerp(p1.normal, p2.normal, t);
    //float2(p1.texcoord.x + t * (p2.texcoord.x - p1.texcoord.x), p1.texcoord.y + t * (p2.texcoord.y - p1.texcoord.y));
    v.texcoord = lerp(p1.texcoord, p2.texcoord, t);
    return v;
}

// Function to clip a triangle against a single clipping plane
static std::vector<Triangle> clipTriangle(const Triangle & triangle, const std::array<float, 4>&plane) {
    std::vector<Vertex> inputVertices;
    std::vector<Vertex> outputVertices;

    inputVertices.push_back(triangle.v0); 
    inputVertices.push_back(triangle.v1);
    inputVertices.push_back(triangle.v2);

    for (size_t i = 0; i < inputVertices.size(); ++i) {
        const Vertex& current = inputVertices[i];
        const Vertex& next = inputVertices[(i + 1) % inputVertices.size()];

        bool currentInside = isInside(current, plane);
        bool nextInside = isInside(next, plane);

        if (currentInside) {
            outputVertices.push_back(current);
        }
        if (currentInside != nextInside) {
            outputVertices.push_back(intersect(current, next, plane));
        }
    }

    // Generate new triangles from the clipped polygon
    std::vector<Triangle> clippedTriangles;
    for (size_t i = 1; i + 1 < outputVertices.size(); ++i) {
        clippedTriangles.push_back({ outputVertices[0], outputVertices[i], outputVertices[i + 1] });
    }

    return clippedTriangles;
}


/*
// perspective project vertices without matrices
void Render::projection(const float3& vertexWorld, const float4x4& worldToCamera, const float& l, const float& r, const float& t, const float& b, 
    const float& nearplane, const uint32_t& imageWidth, const uint32_t& imageHeight, float3& vertexRaster, float3& vertexCamera)
{
    // transform  from world-space to camera-space
    vertexCamera = transform(vertexWorld, worldToCamera);

    // convert to screen-space
    float2 vertexScreen{};
    vertexScreen.x = nearplane * vertexCamera.x / -vertexCamera.z;// negative Z for right-handed coordinate system
    vertexScreen.y = nearplane * vertexCamera.y / -vertexCamera.z;

    // now convert point from screen space to NDC space (in range [-1,1])
    float2 vertexNDC{};
    vertexNDC.x = 2 * vertexScreen.x / (r - l) - (r + l) / (r - l);
    vertexNDC.y = 2 * vertexScreen.y / (t - b) - (t + b) / (t - b);
    
    // convert to raster space
    vertexRaster.x = (vertexNDC.x + 1) / 2 * imageWidth;
    // in raster space y-axis aim down, so invert it`s direction
    vertexRaster.y = (1 - vertexNDC.y) / 2 * imageHeight;
    vertexRaster.z = -vertexCamera.z; // swap z for Right-handed coordinate system only!
}
*/
// perspective project vertices with matrices
/*
void Render::Projection(const XMVECTOR& vertexWorld, 
     const uint32_t& imageWidth, const uint32_t& imageHeight, 
    XMFLOAT3& vertexRaster, XMFLOAT3& vertexCamera)
{
     XMStoreFloat3(&vertexCamera,XMVector3Transform(vertexWorld, XMLoadFloat4x4(&WorldToCamera)));
  
     //float4 vertexNDC = transform(float4(vertexCamera.x, vertexCamera.y, vertexCamera.z,1.0f), CameraToScreen);
     //perspective divide
     //vertexNDC.x /= vertexNDC.w;
     //vertexNDC.y /= vertexNDC.w;
     //vertexNDC.z /= vertexNDC.w;
     //vertexNDC.w /= vertexNDC.w;

     XMVECTOR vertexNDC = XMVector4Transform(XMLoadFloat3(&vertexCamera), XMLoadFloat4x4(&CameraToScreen));
     //perspective divide
     XMVECTOR perspDiv = _mm_shuffle_ps(vertexNDC, vertexNDC, _MM_SHUFFLE(3, 3, 3, 3));
     vertexNDC /= perspDiv;
    
     //Viewport transformation to raster-space
     vertexRaster.x = (vertexNDC.m128_f32[0] + 1) / 2 * imageWidth; //Vextex NDC  X coordinate
     // in raster space y is pointing down so invert it`s direction
     vertexRaster.y = (1 - vertexNDC.m128_f32[1]) / 2 * imageHeight;// Vertex NDC Y coordinate
     vertexRaster.z = -vertexCamera.z; // swap z for Right-handed coordinate system only!
}*/

// perspective project vertices with matrices
void Render::Project(const XMVECTOR& vertexWorld, XMFLOAT3& vertexCamera, XMFLOAT4& vertexNDC)
{
    XMStoreFloat3(&vertexCamera, XMVector3Transform(vertexWorld, XMLoadFloat4x4(&WorldToCamera))); // Model-View transform
    XMStoreFloat4(&vertexNDC ,XMVector3Transform(XMLoadFloat3(&vertexCamera), XMLoadFloat4x4(&CameraToScreen))); // Perspective Projection transform
}

void Render::PerspectiveDivide(const XMFLOAT3& vertexCamera, XMVECTOR& vertexNDC, XMFLOAT3& vertexRaster, const uint32_t& imageWidth,
    const uint32_t& imageHeight)
{
    //perspective divide
    XMVECTOR perspDiv = _mm_shuffle_ps(vertexNDC, vertexNDC, _MM_SHUFFLE(3, 3, 3, 3));
    vertexNDC /= perspDiv;
   
    //Viewport transformation to raster-space
    vertexRaster.x = (vertexNDC.m128_f32[0] + 1) / 2 * imageWidth; //Vextex NDC  X coordinate
    // in raster space y is pointing down so invert it`s direction
    vertexRaster.y = (1 - vertexNDC.m128_f32[1]) / 2 * imageHeight;// Vertex NDC Y coordinate
    vertexRaster.z = -vertexCamera.z; // swap z for Right-handed coordinate system only!
}



void Mesh::ComputeVertexNormals()
{
    size_t numv = vertices.size();
    if (numv == 0 || indices.empty())
        return;

    vector<float3> accum(numv, float3{ 0.0f, 0.0f, 0.0f });
    size_t triCount = indices.size() / 3;
    for (size_t t = 0; t < triCount; ++t)
    {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        float3 v0 = vertices[i0].position;
        float3 v1 = vertices[i1].position;
        float3 v2 = vertices[i2].position;
        float3 N = cross((v1 - v0), (v2 - v0));
        N = normalize(N);
        accum[i0] = accum[i0] + N;
        accum[i1] = accum[i1] + N;
        accum[i2] = accum[i2] + N;
    }

    for (size_t i = 0; i < numv; ++i)
    {
        float3 n = normalize(accum[i]);
        vertices[i].normal = n;
    }
}

// Create a shere with the parametric equation
void Mesh::CreateSphere(float radius, int slices, int stacks)
{
    for (int i = 0; i <= stacks; ++i)// number of latitudes
    {
        // V texture coordinate.
        float V = i / (float)stacks;
        float phi = V * XM_PI; // phi = Pi * latitude step / latitude count

        for (int j = 0; j <= slices; ++j)// number of longitudes
        {
            // U texture coordinate.
            float U = j / (float)slices;
            float theta = U * XM_2PI; // theta = 2 * Pi * longitude step / longitude count

            float X = cos(theta) * sin(phi);
            float Y = cos(phi);
            float Z = sin(theta) * sin(phi);

            Mesh::MeshVertexAoS mv{};
            mv.position = float3(X, Y, Z) * radius;
            mv.normal = float3(X, Y, Z);
            mv.texcoord = float2(U, V);
            mv.tangent = float3(0.0f);
            mv.bitangent = float3(0.0f);
            vertices.push_back(mv);
        }
    }

    for (int i = 0; i < slices * stacks + slices; ++i)
    {
        indices.push_back(i);
        indices.push_back(i + slices + 1);
        indices.push_back(i + slices);

        indices.push_back(i + slices + 1);
        indices.push_back(i);
        indices.push_back(i + 1);
    }
    number_of_triangles = (uint32_t)(indices.size() / 3);
}

void Mesh::CreateSphere2(float radius, int slices, int stacks)
{
  
    // --- VERTEXEK ---

    for (int i = 0; i <= stacks; i++) // szélesség (V)
    {
        float V = i / (float)stacks; // V texture coordinate.
        float phi = V * XM_PI; // 0..pi  phi = Pi * latitude step / latitude count

        for (int j = 0; j <= slices; j++) // hosszúság (U)
        {
            float U = j / (float)slices;  // U texture coordinate.
            float theta = U * XM_2PI; // 0..2pi theta = 2 * Pi * longitude step / longitude count

            float X = radius * std::cos(theta) * std::sin(phi);
            float Y = radius * std::cos(phi);
            float Z = radius * std::sin(theta) * std::sin(phi);

           float3 pos(X, Y, Z);
           float3 n = normalize(pos);
           float2 uv(U, V);

           
            Mesh::MeshVertexAoS mv{};
            mv.position = pos;
            mv.normal = n;
            mv.texcoord = uv;
            mv.tangent = float3(0.0f);
            mv.bitangent = float3(0.0f);
            vertices.push_back(mv);
        }
    }

    // --- INDEXEK ---

    int stride = slices + 1;

    for (int i = 0; i < stacks; i++)
    {
        for (int j = 0; j < slices; j++)
        {
            int i0 = i * stride + j;
            int i1 = i0 + 1;
            int i2 = i0 + stride;
            int i3 = i2 + 1;

            // első háromszög
            indices.push_back(i0);
            indices.push_back(i1);
            indices.push_back(i2);

            // második háromszög
            indices.push_back(i1);
            indices.push_back(i3);
            indices.push_back(i2);
        }
    }
    // texcoordIndices no longer used (AOS vertices store texcoords)
  
    // --- TANGENT / BITANGENT ---
     // calculate the Tangent and Bitangent vectors for the sphere`s triangles	
    float3 edge1{}, edge2{}, tangent{}, bitangent{};
    float2 deltaUV1{}, deltaUV2{};

    const size_t triCount = indices.size() / 3;
    number_of_triangles = (uint32_t)triCount;
    for (size_t t = 0; t < triCount; ++t)
    {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];
        auto& v0 = vertices[i0];
        auto& v1 = vertices[i1];
        auto& v2 = vertices[i2];

        auto& tex0 = v0.texcoord;
        auto& tex1 = v1.texcoord;
        auto& tex2 = v2.texcoord;

        auto& tg0 = v0.tangent;
        auto& tg1 = v1.tangent;
        auto& tg2 = v2.tangent;

        auto& bitg0 = v0.bitangent;
        auto& bitg1 = v1.bitangent;
        auto& bitg2 = v2.bitangent;

        edge1 = v1.position - v0.position;
        edge2 = v2.position - v0.position;
        deltaUV1 = tex1 - tex0;
        deltaUV2 = tex2 - tex0;

        float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        if (std::abs(det) < 1e-8f)
            continue;

        float f = 1.0f / det;

        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

        // itt most összegezzük, hogy simább legyen
        tg0 += tangent;
        tg1 += tangent;
        tg2 += tangent;

        bitg0 += bitangent;
        bitg1 += bitangent;
        bitg2 += bitangent;
    }
    /*
    // opcionális: ortogonalizálás
    for (auto& v : positions)
    {
        float3 n = normalize(v.normal);
        float3 t = v.tangent;

        t = normalize(t - n * dot(n, t));
        float3 b = normalize(cross(n, t));

        v.normal = n;
        v.tangent = t;
        v.bitangent = b;
    }
    */
}

Mesh::MeshSoA Mesh::CreateSphereSoA(uint32_t stacks, uint32_t slices, float radius)
{
    MeshSoA mesh;

    const uint32_t vertCount = (stacks + 1) * (slices + 1);
    mesh.posX.resize(vertCount);
    mesh.posY.resize(vertCount);
    mesh.posZ.resize(vertCount);
    mesh.nrmX.resize(vertCount);
    mesh.nrmY.resize(vertCount);
    mesh.nrmZ.resize(vertCount);
    mesh.uvU.resize(vertCount);
    mesh.uvV.resize(vertCount);

    auto idx = [slices](uint32_t i, uint32_t j)
        {
            return i * (slices + 1) + j;
        };

    // vertexek
    for (uint32_t i = 0; i <= stacks; ++i)
    {
        float v = float(i) / float(stacks);          // [0,1]
        float phi = v * float(XM_PI);                 // [0,pi]

        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);

        for (uint32_t j = 0; j <= slices; ++j)
        {
            float u = float(j) / float(slices);      // [0,1]
            float theta = u * 2.0f * float(XM_PI);    // [0,2pi]

            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            uint32_t k = idx(i, j);

            float nx = sinPhi * cosTheta;
            float ny = cosPhi;
            float nz = sinPhi * sinTheta;

            mesh.nrmX[k] = nx;
            mesh.nrmY[k] = ny;
            mesh.nrmZ[k] = nz;

            mesh.posX[k] = radius * nx;
            mesh.posY[k] = radius * ny;
            mesh.posZ[k] = radius * nz;

            mesh.uvU[k] = u;
            mesh.uvV[k] = 1.0f - v; // klasszikus V flip
        }
    }

    // indexek (triangle list)
    mesh.indices.reserve(stacks * slices * 6);

    for (uint32_t i = 0; i < stacks; ++i)
    {
        for (uint32_t j = 0; j < slices; ++j)
        {
            uint32_t k0 = idx(i, j);
            uint32_t k1 = idx(i + 1, j);
            uint32_t k2 = idx(i + 1, j + 1);
            uint32_t k3 = idx(i, j + 1);

            // két háromszög: (k0,k1,k2) és (k0,k2,k3)
            mesh.indices.push_back(k0);
            mesh.indices.push_back(k1);
            mesh.indices.push_back(k2);

            mesh.indices.push_back(k0);
            mesh.indices.push_back(k2);
            mesh.indices.push_back(k3);
        }
    }

    return mesh;
}

/*
* // 3D vector structure
struct Vec3 {
    double x, y, z;

    // Normalize vector to unit length
    void normalize() {
        double len = std::sqrt(x*x + y*y + z*z);
        if (len > 1e-9) {
            x /= len;
            y /= len;
            z /= len;
        }
    }

    // Midpoint between two vectors
    static Vec3 midpoint(const Vec3& a, const Vec3& b) {
        return { (a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0 };
    }
};

// Triangle made of 3 vertex indices
struct Triangle {
    int v1, v2, v3;
};

// Generate initial octahedron vertices and faces
void createOctahedron(std::vector<Vec3>& vertices, std::vector<Triangle>& faces) {
    vertices = {
        { 1, 0, 0 }, {-1, 0, 0 },
        { 0, 1, 0 }, { 0,-1, 0 },
        { 0, 0, 1 }, { 0, 0,-1 }
    };

    // Normalize to unit sphere
    for (auto& v : vertices) v.normalize();

    faces = {
        {0, 4, 2}, {2, 4, 1}, {1, 4, 3}, {3, 4, 0},
        {0, 2, 5}, {2, 1, 5}, {1, 3, 5}, {3, 0, 5}
    };
}

// Subdivide each triangle into 4 smaller ones
void subdivide(std::vector<Vec3>& vertices, std::vector<Triangle>& faces) {
    std::vector<Triangle> newFaces;
    std::map<std::pair<int,int>, int> midpointCache;

    auto getMidpointIndex = [&](int i1, int i2) -> int {
        auto key = std::minmax(i1, i2);
        if (midpointCache.count(key)) return midpointCache[key];

        Vec3 mid = Vec3::midpoint(vertices[i1], vertices[i2]);
        mid.normalize();
        vertices.push_back(mid);
        int idx = (int)vertices.size() - 1;
        midpointCache[key] = idx;
        return idx;
    };

    for (const auto& tri : faces) {
        int a = getMidpointIndex(tri.v1, tri.v2);
        int b = getMidpointIndex(tri.v2, tri.v3);
        int c = getMidpointIndex(tri.v3, tri.v1);

        newFaces.push_back({tri.v1, a, c});
        newFaces.push_back({tri.v2, b, a});
        newFaces.push_back({tri.v3, c, b});
        newFaces.push_back({a, b, c});
    }

    faces.swap(newFaces);
}

// Generate geosphere from octahedron
void generateGeoSphere(int subdivisions, std::vector<Vec3>& vertices, std::vector<Triangle>& faces) {
    if (subdivisions < 0 || subdivisions > 8) { // limit to avoid huge meshes
        throw std::invalid_argument("Subdivisions must be between 0 and 8.");
    }

    createOctahedron(vertices, faces);
    for (int i = 0; i < subdivisions; ++i) {
        subdivide(vertices, faces);
    }
}

int main() {
    try {
        int subdivisions;
        std::cout << "Enter number of subdivisions (0-8): ";
        if (!(std::cin >> subdivisions)) {
            std::cerr << "Invalid input.\n";
            return 1;
        }

        std::vector<Vec3> vertices;
        std::vector<Triangle> faces;

        generateGeoSphere(subdivisions, vertices, faces);

        std::cout << "Generated geosphere with " 
                  << vertices.size() << " vertices and "
                  << faces.size() << " faces.\n";

        // Example: print first few vertices
        for (size_t i = 0; i < std::min<size_t>(vertices.size(), 10); ++i) {
            std::cout << "v" << i << ": (" 
                      << vertices[i].x << ", " 
                      << vertices[i].y << ", " 
                      << vertices[i].z << ")\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
How It Works
Octahedron Base

6 vertices, 8 triangular faces.
All vertices normalized to lie on the unit sphere.
Subdivision

Each triangle is split into 4 smaller triangles.
Midpoints are cached to avoid duplicate vertices.
Midpoints are normalized to keep them on the sphere.
Complexity

Each subdivision quadruples the number of faces.
Vertex count grows roughly as 
V
≈
2
+
4
⋅
2
n
V≈2+4⋅2 
n
  for 
n
n subdivisions.
Safety

Input validation prevents excessive subdivisions.
Uses std::map to avoid duplicate midpoint vertices.
If you want, I can extend this so it exports the geosphere to an .obj file for use in 3D software or OpenGL.
Would you like me to add that?


*/

/// <summary>
///  Rendering a complete mesh using simple Bounding-box Traversal Rasterization with 1 light-source and procedural pattern.
/// Transform the vertices first and after depth-clipping, render the triangles individually with this method.
/// It use right-handed coordinate system!
/// Compute edge equation incrementally for all pixels.
/// Compute 1 pixel at a time.
/// </summary>
/// <param name="mesh"></param>
void Render::DrawMesh(Mesh* mesh)
{
    XMVECTOR light1_pos = XMLoadFloat3(&light1.position);
    XMVECTOR view_pos = XMLoadFloat3(&viewerPosition);
    // Light`s color
    XMVECTOR Lc = XMLoadFloat4(&light1.color);
    // material`s color
    XMVECTOR Ma = XMLoadFloat4(&mesh->mat.ambient);
    XMVECTOR Md = XMLoadFloat4(&mesh->mat.diffuse);
    XMVECTOR Ms = XMLoadFloat4(&mesh->mat.specular);
    float MaterialPower = mesh->mat.power;
    XMVECTOR Ka = Ma * light1.globalAmbient;

    for (uint32_t i = 0; i < mesh->number_of_triangles; i++)
    {
        std::vector<Vertex> clipped_vertices(3);     
        std::vector<Triangle> clipped_triangles{};
        
        
        // read triangle data
        // read triangle data from AOS vertex storage
        auto& mv0 = mesh->vertices[mesh->indices[i * 3 + 0]];
        auto& mv1 = mesh->vertices[mesh->indices[i * 3 + 1]];
        auto& mv2 = mesh->vertices[mesh->indices[i * 3 + 2]];
        // clip-space position
        clipped_vertices[0].position = mv0.clip_position;
        clipped_vertices[1].position = mv1.clip_position;
        clipped_vertices[2].position = mv2.clip_position;
        // view-space position
        clipped_vertices[0].view_position = mv0.view_position;
        clipped_vertices[1].view_position = mv1.view_position;
        clipped_vertices[2].view_position = mv2.view_position;
        // world-space position
        clipped_vertices[0].world_position = mv0.world_position;
        clipped_vertices[1].world_position = mv1.world_position;
        clipped_vertices[2].world_position = mv2.world_position;
        // normals (world-space)
        clipped_vertices[0].normal = mv0.world_normal;
        clipped_vertices[1].normal = mv1.world_normal;
        clipped_vertices[2].normal = mv2.world_normal;
        // texture coordinates
        clipped_vertices[0].texcoord = mv0.texcoord;
        clipped_vertices[1].texcoord = mv1.texcoord;
        clipped_vertices[2].texcoord = mv2.texcoord;

        std::array<float, 4> plane = { 0, 0, 1, -1 };  // clipping triangles along Z-axis

		clipped_triangles = clipTriangle(Triangle(clipped_vertices[0], clipped_vertices[1], clipped_vertices[2]), plane); // use depth clipping only
       
        for (auto& triangle_begin: clipped_triangles)
        {
        
            XMVECTOR v0 = XMLoadFloat4(&triangle_begin.v0.world_position);  
            XMVECTOR v1 = XMLoadFloat4(&triangle_begin.v1.world_position);
            XMVECTOR v2 = XMLoadFloat4(&triangle_begin.v2.world_position);

            XMVECTOR n0 = XMLoadFloat4(&triangle_begin.v0.normal);
            XMVECTOR n1 = XMLoadFloat4(&triangle_begin.v1.normal);
            XMVECTOR n2 = XMLoadFloat4(&triangle_begin.v2.normal);

            XMVECTOR tex0 = XMLoadFloat2(&triangle_begin.v0.texcoord);
            XMVECTOR tex1 = XMLoadFloat2(&triangle_begin.v1.texcoord);
            XMVECTOR tex2 = XMLoadFloat2(&triangle_begin.v2.texcoord);

            XMFLOAT3 camera_p[3]{}, raster_p[3]{};
            XMVECTOR ndc_p0 = XMLoadFloat4(&triangle_begin.v0.position);  //clip_space_positions
            XMVECTOR ndc_p1 = XMLoadFloat4(&triangle_begin.v1.position);  
            XMVECTOR ndc_p2 = XMLoadFloat4(&triangle_begin.v2.position);  

            XMStoreFloat3(&camera_p[0], XMLoadFloat3(&triangle_begin.v0.view_position)); // view_positions
            XMStoreFloat3(&camera_p[1], XMLoadFloat3(&triangle_begin.v1.view_position));
            XMStoreFloat3(&camera_p[2], XMLoadFloat3(&triangle_begin.v2.view_position));

            PerspectiveDivide(camera_p[0], ndc_p0, raster_p[0], ImageWidth, ImageHeight);
            PerspectiveDivide(camera_p[1], ndc_p1, raster_p[1], ImageWidth, ImageHeight);
            PerspectiveDivide(camera_p[2], ndc_p2, raster_p[2], ImageWidth, ImageHeight);

            XMVECTOR v0Raster = XMLoadFloat4(&XMFLOAT4(raster_p[0].x, raster_p[0].y,raster_p[0].z, 1.0f));
            XMVECTOR v1Raster = XMLoadFloat4(&XMFLOAT4(raster_p[1].x, raster_p[1].y, raster_p[1].z, 1.0f));
            XMVECTOR v2Raster = XMLoadFloat4(&XMFLOAT4(raster_p[2].x, raster_p[2].y, raster_p[2].z, 1.0f));
            
            //compute and normalize the face normal of the current triangle in world-space
            // and use back-face culling
            //XMVECTOR n = XMVector3Cross((v1 - v0), (v2 - v0));
            //n = XMVector3NormalizeEst(n);
            // back-face =  ( v0 - viewer position) dot N >= 0 ,  it is not recommended because this rasterization does automatic back-face culling!
            // if (XMVector3GreaterOrEqual(XMVector3Dot((v0 - XMLoadFloat3(&viewerPosition)), n), XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f))) continue;//draw only front-facing triangles
             
            
            // compute 1/Z
            XMVECTOR v0RasterZ = _mm_rcp_ps(_mm_permute_ps(v0Raster, _MM_PERM_CCCC))
                , v1RasterZ = _mm_rcp_ps(_mm_permute_ps(v1Raster, _MM_SHUFFLE(2, 2, 2, 2)))
                , v2RasterZ = _mm_rcp_ps(_mm_permute_ps(v2Raster, _MM_SHUFFLE(2, 2, 2, 2)));

            // [comment]
            // Precompute reciprocal of vertex z-coordinate
            // [/comment]
             //v0Raster.z = 1 / v0Raster.z,
             //v1Raster.z = 1 / v1Raster.z,
             //v2Raster.z = 1 / v2Raster.z;

            // Prepare vertex attributes to perspective correction. Divide them by their vertex z-coordinate
            // texture coordinates, vertex positions, vertex normals -  x/z, y/z, z/z, w/z
            //tex0 *= v0Raster.z, tex1 *= v1Raster.z, tex2 *= v2Raster.z;
            //v0 *= v0Raster.z; v1 *= v1Raster.z; v2 *= v2Raster.z;
            //n0 *= v0Raster.z; n1 *= v1Raster.z; n2 *= v2Raster.z;

            tex0 *= v0RasterZ, tex1 *= v1RasterZ, tex2 *= v2RasterZ;
            v0 *= v0RasterZ; v1 *= v1RasterZ; v2 *= v2RasterZ;
            n0 *= v0RasterZ; n1 *= v1RasterZ; n2 *= v2RasterZ;

            //float xmin = min3(v0Raster.x, v1Raster.x, v2Raster.x);
            //float ymin = min3(v0Raster.y, v1Raster.y, v2Raster.y);
            //float xmax = max3(v0Raster.x, v1Raster.x, v2Raster.x);
            //float ymax = max3(v0Raster.y, v1Raster.y, v2Raster.y);

            XMVECTOR min = _mm_min_ps(v0Raster, _mm_min_ps(v1Raster, v2Raster));
            XMVECTOR max = _mm_max_ps(v0Raster, _mm_max_ps(v1Raster, v2Raster));
            float xmin = XMVectorGetX(min);
            float xmax = XMVectorGetX(max);
            float ymin = XMVectorGetY(min);
            float ymax = XMVectorGetY(max);
            // eliminate the triangle which is out of screen
            if (xmin > ImageWidth - 1 || xmax < 0 || ymin > ImageHeight - 1 || ymax < 0) continue;

            // compute triangle`s bounding box
            // be careful xmin/xmax/ymin/ymax can be negative. Don't cast to uint32_t
            uint32_t bb_xmin = max(int32_t(0), (int32_t)(std::floor(xmin)));
            uint32_t bb_xmax = min(int32_t(ImageWidth) - 1, (int32_t)(std::floor(xmax)));
            //uint32_t bb_width = bb_xmax - bb_xmin; // BBox width
            uint32_t bb_ymin = max(int32_t(0), (int32_t)(std::floor(ymin)));
            uint32_t bb_ymax = min(int32_t(ImageHeight) - 1, (int32_t)(std::floor(ymax)));
            //uint32_t bb_height = bb_ymax - bb_ymin; // BBox height

            // calculate triangle`s area with scalar values
            //float area = (v2Raster.x - v0Raster.x) * (v1Raster.y - v0Raster.y) - (v2Raster.y - v0Raster.y) * (v1Raster.x - v0Raster.x); //compute triangle`s area
            //if (area <= 0.0f)continue;

            
            XMVECTOR zero_vector = _mm_setzero_ps();

            // calculate 1/area  (the reciprocal of the triangle`s area ) with edge equation
            //generate vectors to calculate triangle`s area 
            XMVECTOR v2x_v2y_v2x_v2y = _mm_permute_ps(v2Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
            XMVECTOR v0x_v0y_v0x_v0y = _mm_permute_ps(v0Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
            XMVECTOR v0x_v0y_0_0 = _mm_shuffle_ps(v0x_v0y_v0x_v0y, zero_vector, _MM_SHUFFLE(0, 0, 1, 0)); // helper vector = | v0x v0y 0 0 |

            XMVECTOR v1y_v1x_v1y_v1x = _mm_permute_ps(v1Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA = yxyx
            XMVECTOR v0y_v0x_v0y_v0x = _mm_permute_ps(v0Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA= yxyx

            //generate vectors for the coefficients of the edge equation  dY12, dY20, dY01 etc.
            XMVECTOR v2y_v0y_v2x_v0x = _mm_permute_ps(_mm_unpacklo_ps(v0Raster, v2Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
            XMVECTOR v1y_v2y_v1x_v2x = _mm_permute_ps(_mm_unpacklo_ps(v2Raster, v1Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
            //the differences are used in edge equation:  dY12=v2y-v1y , dY20=v0y-v2y, dX12=v2x-v1x, dX20=v0x-v2x
            XMVECTOR dY12_dY20_dX12_dX20 = v2y_v0y_v2x_v0x - v1y_v2y_v1x_v2x;
            // dY01 = v1y-v0y , dX01=v1x-v0x
            XMVECTOR dY01_dX01_dY01_dX01 = v1y_v1x_v1y_v1x - v0y_v0x_v0y_v0x;
            //dX02 and dY02 are used for area calculation only!
            XMVECTOR dX02_dY02_dX02_dY02 = v2x_v2y_v2x_v2y - v0x_v0y_v0x_v0y;

            //XMVECTOR _dY01_dX01_dY01_dX01 = dY01_dX01_dY01_dX01;

            //calculate the reciprocal of area (v2x-v0x)*(v1y-v0y) - (v2y-v0y)*(v1x-v0x)
            XMVECTOR dxdy = _mm_mul_ps(dX02_dY02_dX02_dY02, dY01_dX01_dY01_dX01);
            XMVECTOR rcp_area = _mm_rcp_ps(_mm_hsub_ps(dxdy, dxdy));
            rcp_area = _mm_insert_ps(rcp_area, zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0));  // reciprocal area vector | xyzw | =  | 1/area  1/area  1/area  0.0 |


            //calculate edge equation (the determinant) (use right-handed coordinate system!)
            XMVECTOR PxPyPz = XMVectorSet(bb_xmin + 0.5f, bb_ymin + 0.5f, 0.0f, 0.0f);
            //create edge equation`s vectorized version
            XMVECTOR PxPxPx_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(0, 0, 0, 0)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_AAAA = xxxx
            XMVECTOR PyPyPy_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(1, 1, 1, 1)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_BBBB = yyyy
            dY01_dX01_dY01_dX01 = _mm_insert_ps(dY01_dX01_dY01_dX01, zero_vector, _MM_MK_INSERTPS_NDX(0, 1, 4)); // the new vector:  | d01y 0 0 d01x |
            XMVECTOR dY12_dY20_dY01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(1, 0, 1, 0)); //ABAB
            XMVECTOR dX12_dX20_dX01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(2, 3, 3, 2));//CDDC
            XMVECTOR v1x_v2x_v0x_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 0, 3, 2)); // | v1x v2x v0x 0 |
            XMVECTOR v1y_v2y_v0y_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 1, 1, 0)); //  | v1y v2y v0y 0 |
            //XMVECTOR w_0_w_1_w_2 = _mm_fnmadd_ps(dX12_dX20_dX01_0, _mm_sub_ps(PyPyPy_0, v1y_v2y_v0y_0), _mm_mul_ps(dY12_dY20_dY01_0, _mm_sub_ps(PxPxPx_0, v1x_v2x_v0x_0)));
            XMVECTOR w_0_w_1_w_2 = (PxPxPx_0 - v1x_v2x_v0x_0) * dY12_dY20_dY01_0 - (PyPyPy_0 - v1y_v2y_v0y_0) * dX12_dX20_dX01_0;

            //make vector  of  | v0z v1z v2z 0 | for depth testing
            XMVECTOR v0z_v1z_v2z_0 = _mm_insert_ps(v0RasterZ, v1RasterZ, _MM_MK_INSERTPS_NDX(0, 1, 12));
            v0z_v1z_v2z_0 = _mm_insert_ps(v0z_v1z_v2z_0, v2RasterZ, _MM_MK_INSERTPS_NDX(0, 2, 0));


            // extract the coefficients from the edge equation
            //float w_0 = XMVectorGetX(w_0_w_1_w_2);
            //float w_1 = XMVectorGetY(w_0_w_1_w_2);
            //float w_2 = XMVectorGetZ(w_0_w_1_w_2);
            //float dY12 = XMVectorGetX(dY12_dY20_dX12_dX20);
            //float dX12= XMVectorGetZ(dY12_dY20_dX12_dX20);
            //float dY20= XMVectorGetY(dY12_dY20_dX12_dX20);
            //float dX20= XMVectorGetW(dY12_dY20_dX12_dX20);
            //float dY01 = XMVectorGetX(dY01_dX01_dY01_dX01);
            //float dX01 = XMVectorGetY(_dY01_dX01_dY01_dX01);

            // scalar code for calculate edge equation
            //XMFLOAT3 P(x0 + 0.5, y0 + 0.5, 0);
            //float dY12 = v2Raster.y - v1Raster.y , dX12 = v2Raster.x - v1Raster.x;
            //float dY20 = v0Raster.y - v2Raster.y, dX20 = v0Raster.x - v2Raster.x;
            //float dY01 = v1Raster.y - v0Raster.y, dX01 = v1Raster.x - v0Raster.x;
            //float w_0 = (P.x - v1Raster.x) * dY12 - (P.y - v1Raster.y) * dX12;//edge function (v1Raster, v2Raster, P);
            //float w_1 = (P.x - v2Raster.x) * dY20 - (P.y - v2Raster.y) * dX20;//edge function (v2Raster, v0Raster, P);
            //float w_2 = (P.x - v0Raster.x) * dY01 - (P.y - v0Raster.y) * dX01;//edge function (v0Raster, v1Raster, P);

            // [comment]
            // check the triangle`s bounding area only, render this area of the screen
            // [/comment]
            for (uint32_t y = bb_ymin; y <= bb_ymax; y++)
            {
                //float w0 = w_0 , w1 = w_1 , w2 = w_2;
                XMVECTOR w0_w1_w2 = w_0_w_1_w_2;
                for (uint32_t x = bb_xmin; x <= bb_xmax; x++)
                {

                    //XMFLOAT3 P(x + 0.5, y + 0.5, 0);
                    // edge-functions for right-handed coordinate system
                     //  the determinants
                    //      | Px - v0x   v1x - v0x |            | Px - v2x   v0x - v2x |            | Px - v1x   v2x - v1x |
                    //      | Py - v0y   v1y - v0y |           | Py - v2y   v0y - v2y |            | Py - v1y   v2y - v1y |
                    //
                    //float w0 = (P.x - v1Raster.x) * (v2Raster.y - v1Raster.y) - (P.y - v1Raster.y) * (v2Raster.x - v1Raster.x);   //edge function (v1Raster, v2Raster, P);
                    //float w1 = (P.x - v2Raster.x) * (v0Raster.y - v2Raster.y) - (P.y - v2Raster.y) * (v0Raster.x - v2Raster.x);   /edge function (v2Raster, v0Raster, P);
                    //float w2 = (P.x - v0Raster.x) * (v1Raster.y - v0Raster.y) - (P.y - v0Raster.y) * (v1Raster.x - v0Raster.x);   //edge function (v0Raster, v1Raster, P);
                    //  the following equations are the same as above
                    //float w0 = - (v2Raster.x - v1Raster.x) * (P.y - v1Raster.y) + (v2Raster.y - v1Raster.y) * (P.x - v1Raster.x);
                    //float w1 = - (v0Raster.x - v2Raster.x) * (P.y - v2Raster.y) + (v0Raster.y - v2Raster.y) * (P.x - v2Raster.x);
                    //float w2 = - (v1Raster.x - v0Raster.x) * (P.y - v0Raster.y) + (v1Raster.y - v0Raster.y) * (P.x - v0Raster.x);


                    //edge-functions for left-handed coordinate system
                    //  the determinants
                    //      | v1x - v0x   Px - v0x |        | v0x - v2x     Px - v2x |       | v2x - v1x     Px - v1x |             
                    //      | v1y - v0y   Py - v0y |       | v0y - v2y     Py - v2y |       | v2y - v1y     Py - v1y |             
                    // 
                    //float w0 = (v2Raster.x - v1Raster.x) * (P.y - v1Raster.y) - (v2Raster.y - v1Raster.y) * (P.x - v1Raster.x);
                    //float w1 = (v0Raster.x - v2Raster.x) * (P.y - v2Raster.y) - (v0Raster.y - v2Raster.y) * (P.x - v2Raster.x);
                    //float w2 = (v1Raster.x - v0Raster.x) * (P.y - v0Raster.y) - (v1Raster.y - v0Raster.y) * (P.x - v0Raster.x);

                    // or use above equation with right-handed transformation (the sign of the equation is flipped), but the if statement need to be changed to : if (  w0 <= 0 && w1 <= 0 && w2 <= 0  )
                    // check if  right-handed is used
                    //XMVECTOR c = _mm_cmpge_ps(w0_w1_w2, zero_vector); // this comparison is not necessary , w0 && w1 && w2 >=0
                    //XMVECTOR cmp = _mm_insert_ps(c, zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0));
                     /* w0 >= 0 && w1 >= 0 && w2 >= 0 */ /*((int)w0 | (int)w1 | (int)w2) >= 0*/

                    int mask = _mm_movemask_ps(w0_w1_w2); // check the sign bit of the weights (edge equation values) and if they are zero, than the pixel is part of the triangle
                    //!!! if only the first conditional statement is used below then only front-facing triangles will be drawn, otherwise both front- and back-facing triangles will be drawn!
                    if ((mask & 7) == 0 || (mask & 7) == 7) //draw both front- and back-faces
                    {

                        //compute the barycentric coordinates
                        //float bary0 =   w0 / area;
                        //float bary1 =   w1 / area;
                        //float bary2 =   w2 / area;
                        //float bary0 = 1 - bary1 - bary2;

                        XMVECTOR bary0_bary1_bary2_0 = w0_w1_w2 * rcp_area; // barycentric = weight * 1/area

                        //interpolate the depth value
                        //float z = 1 / (v0Raster.z * bary0 + v1Raster.z * bary1 + v2Raster.z * bary2);
                        //another computation: 
                        // 1/z = z0 + bary1 * ( z1 -z0) + bary2 * (z2 - z0)
                        //float z = 1 / v0Raster.z+bary1*(v1Raster.z - v0Raster.z)+bary2*(v2Raster.z - v0Raster.z)
                        //

                        float z = 1 / XMVectorGetX(XMVector3Dot(v0z_v1z_v2z_0, bary0_bary1_bary2_0));
                        z = clamp(z, nearClippingPLane, farClippingPLane);

                        float bary0 = XMVectorGetX(bary0_bary1_bary2_0);
                        float bary1 = XMVectorGetY(bary0_bary1_bary2_0);
                        float bary2 = XMVectorGetZ(bary0_bary1_bary2_0);
                        //XMVECTOR bary0v = _mm_set1_ps(bary0);
                        //XMVECTOR bary1v = _mm_set1_ps(bary1);
                        //XMVECTOR bary2v = _mm_set1_ps(bary2);
                        // Depth-buffer test
                        if (z < depthBuffer[(uint64_t)y * ImageWidth + x]) {
                            z += 0.001f;  // add a depth-bias 
                            depthBuffer[(uint64_t)y * ImageWidth + x] = z;

                            // vertex attribute interpolation
                            // interpolate the texture coordinate
                            XMVECTOR texcoord = tex0 * bary0 + tex1 * bary1 + tex2 * bary2;
                            //XMVECTOR texcoord = _mm_fmadd_ps(tex0, bary0v, _mm_fmadd_ps(tex1, bary1v, _mm_mul_ps(tex2, bary2v)));
                            texcoord *= z; // perspective correction!
                            
                            //compute the point`s interpolated coordinate in world-space for per-pixel lighting
                            XMVECTOR pt = v0 * bary0 + v1 * bary1 + v2 * bary2;
                            //XMVECTOR pt = _mm_fmadd_ps(v0, bary0v, _mm_fmadd_ps(v1, bary1v, _mm_mul_ps(v2, bary2v)));
                            pt *= z;
                            
                            // compute per-pixel normals
                            XMVECTOR N = n0 * bary0 + n1 * bary1 + n2 * bary2;
                            //XMVECTOR N = _mm_fmadd_ps(n0, bary0v, _mm_fmadd_ps(n1, bary1v, _mm_mul_ps(n2, bary2v)));
                            N *= z;
                            N = XMVector3NormalizeEst(N);

                            //pixel-shader code
                            // shade the pixel by calculate per-pixel direct lighting using a simple point-light                       
                            XMVECTOR L = light1_pos - pt; //  light direction
                            float distance = XMVectorGetX(XMVector3Length(L));
                            L /= distance;
                            float attenuation = 1.0f / (light1.constantAttenuation + light1.linearAttenuation * distance + light1.quadraticAttenuation * distance * distance);
                            XMVECTOR V = view_pos - pt; // viewer direction
                            V = XMVector3NormalizeEst(V);
                            XMVECTOR H = L + V; // halfway vector
                            H = XMVector3NormalizeEst(H);
                            //XMVECTOR R = XMVector3Reflect(-L, N);  // 2* (N dot L)* N - L


                            XMVECTOR NdotL = XMVectorMax(XMVector3Dot(N, L), zero_vector);
                            XMVECTOR NdotH = XMVectorMax(XMVector3Dot(N, H), zero_vector); //blinn-phong brdf
                            //XMVECTOR RdotV = XMVectorMax( XMVector3Dot(R, V),zero_vector);// phong brdf

                            // diffuse coefficient
                            XMVECTOR Kd = NdotL * (Lc * Md) * attenuation; 
                            // specular coefficient
                            XMVECTOR Ks = pow(XMVectorGetX(NdotH), MaterialPower) * (Lc * Ms) * attenuation;   // exp(MaterialPower*log(NdotH))  

                            XMVECTOR finalcolor = XMVectorSaturate((Ka + Kd + Ks));


                            // [comment]
                            // The final color is the result of the faction ration multiplied by the
                            // checkerboard pattern.
                            // [/comment]
                            constexpr int M = 8;
                            float checker = (fmod(XMVectorGetX(texcoord) * 6, 1.0f) > 0.2f) ^ (fmod(XMVectorGetY(texcoord) * 20, 1.0f) < 0.3f);
                            float pattern = 0.9f * (1.0f - checker) + 1.0f * checker;
                            
                            /*
                              float scaleS = 5, scaleT = 5; // scale of the pattern
                               inline float modulo(const float &x) 
                                { 
                                    return x - std::floor(x); 
                                } 
... 
                                float pattern = (modulo(XMVectorGetY(texcoord) * scaleT) < 0.5) ^ (modulo(XMVectorGetX(texcoord) * scaleS) < 0.5); 
                            */
                             //float scaleS = 5, scaleT = 5; // scale of the pattern
                             //float pattern = (cos(XMVectorGetY(texcoord) * 2 * XM_PI * scaleT) * sin(XMVectorGetX(texcoord) * 2 * XM_PI * scaleS) + 1) * 0.5; // compute sine wave pattern
                            
                            finalcolor *= pattern;
                            finalcolor *= 255.0f; //convert pixel color from [0.0,1.0] to [0,255] range                    
                            frameBuffer[(uint64_t)y * ImageWidth + x] = finalcolor;
							//if (bary0 < 0.009 || bary1 < 0.009 || bary2 < 0.009) 
                                //frameBuffer[(uint64_t)y * ImageWidth + x] = BGRA(255,255,255,255);

                        }

                    }
                    //edge equation: E(X, Y) = (px - X) * dY - (py - Y) * dX     p(x,y) = pixel position inside the bounding-box
                    //inner-loop  , step to the next pixel in the current row
                    // calculate the determinant (edge-equation) incrementally for all pixels within the bounding-box
                    //incremental equation (x-axis): E(X + 1, Y) = E(X, Y) + dY
                    //w0 += dY12; w1 += dY20; w2 += dY01;
                    w0_w1_w2 += dY12_dY20_dY01_0;
                }
                //outer-loop , step down to the next row
                //incremental equation (y-axis): E(X, Y + 1) =  E (X, Y) - dX
                //w_0 -= dX12; w_1 -= dX20; w_2 -= dX01;          
                w_0_w_1_w_2 -= dX12_dX20_dX01_0;
            }
        }
    }
}

// Tile-based parallel renderer: split vertical screen into N tiles (N = logical processors)

void Render::DrawMeshTiled(Mesh* mesh)
{
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    const uint32_t imageW = ImageWidth;
    const uint32_t imageH = ImageHeight;

    // compute tile heights (last tile may be slightly larger)
    uint32_t baseTileH = imageH / numThreads;
    uint32_t rem = imageH % numThreads;

    // Launch worker threads
    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
        const uint32_t tileY0 = tid * baseTileH + std::min<uint32_t>(tid, rem);
        const uint32_t tileH = baseTileH + (tid < rem ? 1u : 0u);
        const uint32_t tileY1 = tileY0 + tileH - 1;

        workers.emplace_back([=]()
            {
                // Each worker iterates triangles and rasterizes only pixels inside its tile
                XMVECTOR light1_pos = XMLoadFloat3(&light1.position);
                XMVECTOR view_pos = XMLoadFloat3(&viewerPosition);
                // Light`s color
                XMVECTOR Lc = XMLoadFloat4(&light1.color);
                // material`s color
                XMVECTOR Ma = XMLoadFloat4(&mesh->mat.ambient);
                XMVECTOR Md = XMLoadFloat4(&mesh->mat.diffuse);
                XMVECTOR Ms = XMLoadFloat4(&mesh->mat.specular);
                float MaterialPower = mesh->mat.power;
                XMVECTOR Ka = Ma * light1.globalAmbient;

                for (uint32_t i = 0; i < mesh->number_of_triangles; i++)
                {
                    std::vector<Vertex> clipped_vertices{};
                    clipped_vertices.push_back(Vertex());
                    clipped_vertices.push_back(Vertex());
                    clipped_vertices.push_back(Vertex());
                    std::vector<Triangle> clipped_triangles{};


                    // read triangle data
                    // read triangle data from AOS vertex storage
                    auto& mv0 = mesh->vertices[mesh->indices[i * 3 + 0]];
                    auto& mv1 = mesh->vertices[mesh->indices[i * 3 + 1]];
                    auto& mv2 = mesh->vertices[mesh->indices[i * 3 + 2]];
                    // clip-space position
                    clipped_vertices[0].position = mv0.clip_position;
                    clipped_vertices[1].position = mv1.clip_position;
                    clipped_vertices[2].position = mv2.clip_position;
                    // view-space position
                    clipped_vertices[0].view_position = mv0.view_position;
                    clipped_vertices[1].view_position = mv1.view_position;
                    clipped_vertices[2].view_position = mv2.view_position;
                    // world-space position
                    clipped_vertices[0].world_position = mv0.world_position;
                    clipped_vertices[1].world_position = mv1.world_position;
                    clipped_vertices[2].world_position = mv2.world_position;
                    // normals (world-space)
                    clipped_vertices[0].normal = mv0.world_normal;
                    clipped_vertices[1].normal = mv1.world_normal;
                    clipped_vertices[2].normal = mv2.world_normal;
                    // texture coordinates
                    clipped_vertices[0].texcoord = mv0.texcoord;
                    clipped_vertices[1].texcoord = mv1.texcoord;
                    clipped_vertices[2].texcoord = mv2.texcoord;

                    std::array<float, 4> plane = { 0, 0, 1, -1 };  // clipping triangles along Z-axis

                    clipped_triangles = clipTriangle(Triangle(clipped_vertices[0], clipped_vertices[1], clipped_vertices[2]), plane); // use depth clipping only

                    for (auto& triangle_begin : clipped_triangles)
                    {

                        XMVECTOR v0 = XMLoadFloat4(&triangle_begin.v0.world_position);
                        XMVECTOR v1 = XMLoadFloat4(&triangle_begin.v1.world_position);
                        XMVECTOR v2 = XMLoadFloat4(&triangle_begin.v2.world_position);

                        XMVECTOR n0 = XMLoadFloat4(&triangle_begin.v0.normal);
                        XMVECTOR n1 = XMLoadFloat4(&triangle_begin.v1.normal);
                        XMVECTOR n2 = XMLoadFloat4(&triangle_begin.v2.normal);

                        XMVECTOR tex0 = XMLoadFloat2(&triangle_begin.v0.texcoord);
                        XMVECTOR tex1 = XMLoadFloat2(&triangle_begin.v1.texcoord);
                        XMVECTOR tex2 = XMLoadFloat2(&triangle_begin.v2.texcoord);

                        XMFLOAT3 camera_p[3]{}, raster_p[3]{};
                        XMVECTOR ndc_p0 = XMLoadFloat4(&triangle_begin.v0.position);  //clip_space_positions
                        XMVECTOR ndc_p1 = XMLoadFloat4(&triangle_begin.v1.position);
                        XMVECTOR ndc_p2 = XMLoadFloat4(&triangle_begin.v2.position);

                        XMStoreFloat3(&camera_p[0], XMLoadFloat3(&triangle_begin.v0.view_position)); // view_positions
                        XMStoreFloat3(&camera_p[1], XMLoadFloat3(&triangle_begin.v1.view_position));
                        XMStoreFloat3(&camera_p[2], XMLoadFloat3(&triangle_begin.v2.view_position));

                        PerspectiveDivide(camera_p[0], ndc_p0, raster_p[0], imageW, imageH);
                        PerspectiveDivide(camera_p[1], ndc_p1, raster_p[1], imageW, imageH);
                        PerspectiveDivide(camera_p[2], ndc_p2, raster_p[2], imageW, imageH);

                        XMVECTOR v0Raster = XMLoadFloat4(&XMFLOAT4(raster_p[0].x, raster_p[0].y, raster_p[0].z, 1.0f));
                        XMVECTOR v1Raster = XMLoadFloat4(&XMFLOAT4(raster_p[1].x, raster_p[1].y, raster_p[1].z, 1.0f));
                        XMVECTOR v2Raster = XMLoadFloat4(&XMFLOAT4(raster_p[2].x, raster_p[2].y, raster_p[2].z, 1.0f));

                        // compute 1/Z
                        XMVECTOR v0RasterZ = _mm_rcp_ps(_mm_permute_ps(v0Raster, _MM_PERM_CCCC))
                            , v1RasterZ = _mm_rcp_ps(_mm_permute_ps(v1Raster, _MM_SHUFFLE(2, 2, 2, 2)))
                            , v2RasterZ = _mm_rcp_ps(_mm_permute_ps(v2Raster, _MM_SHUFFLE(2, 2, 2, 2)));

                  
                        // Precompute reciprocal of vertex z-coordinate
                        // Prepare vertex attributes to perspective correction. Divide them by their vertex z-coordinate
                        // texture coordinates, vertex positions, vertex normals -  x/z, y/z, z/z, w/z

                        tex0 *= v0RasterZ, tex1 *= v1RasterZ, tex2 *= v2RasterZ;
                        v0 *= v0RasterZ; v1 *= v1RasterZ; v2 *= v2RasterZ;
                        n0 *= v0RasterZ; n1 *= v1RasterZ; n2 *= v2RasterZ;
                       
                        XMVECTOR min = _mm_min_ps(v0Raster, _mm_min_ps(v1Raster, v2Raster));
                        XMVECTOR max = _mm_max_ps(v0Raster, _mm_max_ps(v1Raster, v2Raster));
                        float xmin = XMVectorGetX(min);
                        float xmax = XMVectorGetX(max);
                        float ymin = XMVectorGetY(min);
                        float ymax = XMVectorGetY(max);
                        // eliminate the triangle which is out of screen
                        if (xmin > imageW - 1 || xmax < 0 || ymin > imageH - 1 || ymax < 0) continue;

                        // compute triangle`s bounding box
                        // be careful xmin/xmax/ymin/ymax can be negative. Don't cast to uint32_t
                        uint32_t bb_xmin = max(int32_t(0), (int32_t)(std::floor(xmin)));
                        uint32_t bb_xmax = min(int32_t(imageW) - 1, (int32_t)(std::floor(xmax)));
                        uint32_t bb_ymin = max(int32_t(0), (int32_t)(std::floor(ymin)));
                        uint32_t bb_ymax = min(int32_t(imageH) - 1, (int32_t)(std::floor(ymax)));
                        // intersect with this thread's tile Y-range
                        if (bb_ymax < (int32_t)tileY0 || bb_ymin >(int32_t)tileY1) continue;
                        bb_ymin = std::max<int32_t>(bb_ymin, (int32_t)tileY0);
                        bb_ymax = std::min<int32_t>(bb_ymax, (int32_t)tileY1);


                        XMVECTOR zero_vector = _mm_setzero_ps();
                        // calculate 1/area  (the reciprocal of the triangle`s area ) with edge equation
                        //generate vectors to calculate triangle`s area 
                        XMVECTOR v2x_v2y_v2x_v2y = _mm_permute_ps(v2Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
                        XMVECTOR v0x_v0y_v0x_v0y = _mm_permute_ps(v0Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
                        XMVECTOR v0x_v0y_0_0 = _mm_shuffle_ps(v0x_v0y_v0x_v0y, zero_vector, _MM_SHUFFLE(0, 0, 1, 0)); // helper vector = | v0x v0y 0 0 |

                        XMVECTOR v1y_v1x_v1y_v1x = _mm_permute_ps(v1Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA = yxyx
                        XMVECTOR v0y_v0x_v0y_v0x = _mm_permute_ps(v0Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA= yxyx

                        //generate vectors for the coefficients of the edge equation  dY12, dY20, dY01 etc.
                        XMVECTOR v2y_v0y_v2x_v0x = _mm_permute_ps(_mm_unpacklo_ps(v0Raster, v2Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
                        XMVECTOR v1y_v2y_v1x_v2x = _mm_permute_ps(_mm_unpacklo_ps(v2Raster, v1Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
                        //the differences are used in edge equation:  dY12=v2y-v1y , dY20=v0y-v2y, dX12=v2x-v1x, dX20=v0x-v2x
                        XMVECTOR dY12_dY20_dX12_dX20 = v2y_v0y_v2x_v0x - v1y_v2y_v1x_v2x;
                        // dY01 = v1y-v0y , dX01=v1x-v0x
                        XMVECTOR dY01_dX01_dY01_dX01 = v1y_v1x_v1y_v1x - v0y_v0x_v0y_v0x;
                        //dX02 and dY02 are used for area calculation only!
                        XMVECTOR dX02_dY02_dX02_dY02 = v2x_v2y_v2x_v2y - v0x_v0y_v0x_v0y;

                        //XMVECTOR _dY01_dX01_dY01_dX01 = dY01_dX01_dY01_dX01;

                        //calculate the reciprocal of area (v2x-v0x)*(v1y-v0y) - (v2y-v0y)*(v1x-v0x)
                        XMVECTOR dxdy = _mm_mul_ps(dX02_dY02_dX02_dY02, dY01_dX01_dY01_dX01);
                        XMVECTOR rcp_area = _mm_rcp_ps(_mm_hsub_ps(dxdy, dxdy));
                        rcp_area = _mm_insert_ps(rcp_area, zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0));  // reciprocal area vector | xyzw | =  | 1/area  1/area  1/area  0.0 |


                        //calculate edge equation (the determinant) (use right-handed coordinate system!)
                        XMVECTOR PxPyPz = XMVectorSet(bb_xmin + 0.5f, bb_ymin + 0.5f, 0.0f, 0.0f);
                        //create edge equation`s vectorized version
                        XMVECTOR PxPxPx_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(0, 0, 0, 0)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_AAAA = xxxx
                        XMVECTOR PyPyPy_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(1, 1, 1, 1)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_BBBB = yyyy
                        dY01_dX01_dY01_dX01 = _mm_insert_ps(dY01_dX01_dY01_dX01, zero_vector, _MM_MK_INSERTPS_NDX(0, 1, 4)); // the new vector:  | d01y 0 0 d01x |
                        XMVECTOR dY12_dY20_dY01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(1, 0, 1, 0)); //ABAB
                        XMVECTOR dX12_dX20_dX01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(2, 3, 3, 2));//CDDC
                        XMVECTOR v1x_v2x_v0x_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 0, 3, 2)); // | v1x v2x v0x 0 |
                        XMVECTOR v1y_v2y_v0y_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 1, 1, 0)); //  | v1y v2y v0y 0 |
                        //XMVECTOR w_0_w_1_w_2 = _mm_fnmadd_ps(dX12_dX20_dX01_0, _mm_sub_ps(PyPyPy_0, v1y_v2y_v0y_0), _mm_mul_ps(dY12_dY20_dY01_0, _mm_sub_ps(PxPxPx_0, v1x_v2x_v0x_0)));
                        XMVECTOR w_0_w_1_w_2 = (PxPxPx_0 - v1x_v2x_v0x_0) * dY12_dY20_dY01_0 - (PyPyPy_0 - v1y_v2y_v0y_0) * dX12_dX20_dX01_0;

                        //make vector  of  | v0z v1z v2z 0 | for depth testing
                        XMVECTOR v0z_v1z_v2z_0 = _mm_insert_ps(v0RasterZ, v1RasterZ, _MM_MK_INSERTPS_NDX(0, 1, 12));
                        v0z_v1z_v2z_0 = _mm_insert_ps(v0z_v1z_v2z_0, v2RasterZ, _MM_MK_INSERTPS_NDX(0, 2, 0));

                        // check the triangle`s bounding area only, render this area of the screen
                       // edge equation: w_0_w_1_w_2 a (bb_xmin + 0.5, bb_ymin + 0.5) pontra érvényes
                        // dY12_dY20_dY01_0, dX12_dX20_dX01_0 már ki van számolva

                        constexpr uint32_t TILE = 16;

                        // tile-okra bontjuk a bounding-boxot
                        for (uint32_t tileY = bb_ymin; tileY <= bb_ymax; tileY += TILE)
                        {
                            uint32_t tileYEnd = min(tileY + TILE - 1, bb_ymax);

                            for (uint32_t tileX = bb_xmin; tileX <= bb_xmax; tileX += TILE)
                            {
                                uint32_t tileXEnd = min(tileX + TILE - 1, bb_xmax);

                                // ha ez a thread csak tileY0..tileY1 sávot rajzolhat, akkor itt is szűrünk:
                                if (tileYEnd < tileY0 || tileY > tileY1)
                                    continue;

                                uint32_t yStart = max(tileY, tileY0);
                                uint32_t yEnd = min(tileYEnd, tileY1);

                                // deltaX, deltaY a bounding-box bal-felső sarkához képest
                                int32_t deltaX = (int32_t)tileX - (int32_t)bb_xmin;
                                int32_t deltaY = (int32_t)tileY - (int32_t)bb_ymin;

                                // w érték a tile bal-felső pixelén (tileX, tileY)
                                // E(X+dx, Y+dy) = E(X,Y) + dY*dx - dX*dy
                                XMVECTOR w_tile_row_start =
                                    w_0_w_1_w_2
                                    + dY12_dY20_dY01_0 * _mm_set1_ps((float)deltaX)
                                    - dX12_dX20_dX01_0 * _mm_set1_ps((float)deltaY);

                                // végigmegyünk a tile sorain
                                XMVECTOR w_row_start = w_tile_row_start;
                                
                                for (uint32_t y = yStart; y <= yEnd; ++y)
                                {
                                    XMVECTOR w0_w1_w2 = w_row_start;

                                    for (uint32_t x = tileX; x <= tileXEnd; ++x)
                                    {
                                        int mask = _mm_movemask_ps(w0_w1_w2);
                                        if ((mask & 7) == 0 || (mask & 7) == 7) // front + back faces
                                        {
                                            XMVECTOR bary0_bary1_bary2_0 = w0_w1_w2 * rcp_area;

                                            float z = 1.0f / XMVectorGetX(XMVector3Dot(v0z_v1z_v2z_0, bary0_bary1_bary2_0));
                                            z = clamp(z, nearClippingPLane, farClippingPLane);

                                            float bary0 = XMVectorGetX(bary0_bary1_bary2_0);
                                            float bary1 = XMVectorGetY(bary0_bary1_bary2_0);
                                            float bary2 = XMVectorGetZ(bary0_bary1_bary2_0);

                                            // pixel shading + depth test
                                            uint64_t idx = (uint64_t)y * imageW + x;
                                            if (z < depthBuffer[idx])
                                            {
                                                z += 0.001f;
                                                depthBuffer[idx] = z;
                                                

                                                //----------------------------------- shader-kod
                                                XMVECTOR texcoord = tex0 * bary0 + tex1 * bary1 + tex2 * bary2;
                                                texcoord *= z;

                                                XMVECTOR pt = v0 * bary0 + v1 * bary1 + v2 * bary2;
                                                pt *= z;

                                                XMVECTOR N = n0 * bary0 + n1 * bary1 + n2 * bary2;
                                                N *= z;
                                                N = XMVector3NormalizeEst(N);

                                                XMVECTOR L = light1_pos - pt;
                                                float distance = XMVectorGetX(XMVector3Length(L));
                                                L /= distance;
                                                float attenuation = 1.0f / (light1.constantAttenuation + light1.linearAttenuation * distance + light1.quadraticAttenuation * distance * distance);

                                                XMVECTOR V = view_pos - pt;
                                                V = XMVector3NormalizeEst(V);
                                                XMVECTOR H = XMVector3NormalizeEst(L + V);

                                                XMVECTOR NdotL = XMVectorMax(XMVector3Dot(N, L), zero_vector);
                                                XMVECTOR NdotH = XMVectorMax(XMVector3Dot(N, H), zero_vector);

                                                XMVECTOR Kd = NdotL * (Lc * Md) * attenuation;
                                                XMVECTOR Ks = pow(XMVectorGetX(NdotH), MaterialPower) * (Lc * Ms) * attenuation;

                                                XMVECTOR finalcolor = XMVectorSaturate((Ka + Kd  + Ks));
                                                
                                                float checker = (fmod(XMVectorGetX(texcoord) * 6, 1.0f) > 0.2f)
                                                    ^ (fmod(XMVectorGetY(texcoord) * 20, 1.0f) < 0.3f);
                                                float pattern = 0.9f * (1.0f - checker) + 1.0f * checker;
                                                //------------------------------------------ shader-kod vege

                                                finalcolor *= pattern;
                                                //XMVECTOR finalcolor = _mm_set_ps(1.0f, 0.5f, 0.2f, 1.0f);
                                                finalcolor *= 255.0f;

                                                frameBuffer[idx] = finalcolor;
                                            }
                                           
                                        }

                                        // x irányú inkrementálás: E(X+1,Y) = E(X,Y) + dY
                                        w0_w1_w2 += dY12_dY20_dY01_0;
                                    }

                                    // y irányú inkrementálás a következő sorhoz: E(X,Y+1) = E(X,Y) - dX
                                    w_row_start -= dX12_dX20_dX01_0;
                                }
                            }
                        }

                    }
                }
            });// end of worker thread
    }
    // join threads
    for (auto& t : workers) if (t.joinable()) t.join();
}

/// <summary>
///Tiled Draw Mesh with triangle-binning
/// </summary>
/// <param name="mesh"></param>
void Render::DrawMeshTiledBinning(Mesh* mesh)
{
    const uint32_t imageW = ImageWidth;
    const uint32_t imageH = ImageHeight;

    const int TILE_SIZE = 32;
    const int tilesX = (imageW + TILE_SIZE - 1) / TILE_SIZE;
    const int tilesY = (imageH + TILE_SIZE - 1) / TILE_SIZE;
   

    struct TileBin
    {
        std::vector<uint32_t> triIndices; // index TriWork-ba
    };

    std::vector<TriWork> triWorkList;
    triWorkList.reserve(mesh->number_of_triangles * 2); // clipping miatt lehet több

    std::vector<TileBin> bins(tilesX * tilesY);

    // --------------------------- Binning-pass begin
    for (uint32_t i = 0; i < mesh->number_of_triangles; i++)
    {
        std::vector<Vertex> clipped_vertices(3);
        std::vector<Triangle> clipped_triangles{};

        // read triangle data
        // read triangle data from AOS vertex storage
        auto& mv0 = mesh->vertices[mesh->indices[i * 3 + 0]];
        auto& mv1 = mesh->vertices[mesh->indices[i * 3 + 1]];
        auto& mv2 = mesh->vertices[mesh->indices[i * 3 + 2]];
        // clip-space position
        clipped_vertices[0].position = mv0.clip_position;
        clipped_vertices[1].position = mv1.clip_position;
        clipped_vertices[2].position = mv2.clip_position;
        // view-space position
        clipped_vertices[0].view_position = mv0.view_position;
        clipped_vertices[1].view_position = mv1.view_position;
        clipped_vertices[2].view_position = mv2.view_position;
        // world-space position
        clipped_vertices[0].world_position = mv0.world_position;
        clipped_vertices[1].world_position = mv1.world_position;
        clipped_vertices[2].world_position = mv2.world_position;
        // normals (world-space)
        clipped_vertices[0].normal = mv0.world_normal;
        clipped_vertices[1].normal = mv1.world_normal;
        clipped_vertices[2].normal = mv2.world_normal;
        // texture coordinates
        clipped_vertices[0].texcoord = mv0.texcoord;
        clipped_vertices[1].texcoord = mv1.texcoord;
        clipped_vertices[2].texcoord = mv2.texcoord;

        std::array<float, 4> plane = { 0, 0, 1, -1 };  // clipping triangles along Z-axis

        clipped_triangles = clipTriangle(Triangle(clipped_vertices[0], clipped_vertices[1], clipped_vertices[2]), plane); // use depth clipping only

        for (auto& triangle_begin : clipped_triangles)
        {
            // v0,v1,v2, n0,n1,n2, tex0,tex1,tex2, camera_p, raster_p, v0Raster,v1Raster,v2Raster,
            // v0RasterZ,v1RasterZ,v2RasterZ – pontosan ugyanúgy, mint most
            XMVECTOR v0 = XMLoadFloat4(&triangle_begin.v0.world_position);
            XMVECTOR v1 = XMLoadFloat4(&triangle_begin.v1.world_position);
            XMVECTOR v2 = XMLoadFloat4(&triangle_begin.v2.world_position);

            XMVECTOR n0 = XMLoadFloat4(&triangle_begin.v0.normal);
            XMVECTOR n1 = XMLoadFloat4(&triangle_begin.v1.normal);
            XMVECTOR n2 = XMLoadFloat4(&triangle_begin.v2.normal);

            XMVECTOR tex0 = XMLoadFloat2(&triangle_begin.v0.texcoord);
            XMVECTOR tex1 = XMLoadFloat2(&triangle_begin.v1.texcoord);
            XMVECTOR tex2 = XMLoadFloat2(&triangle_begin.v2.texcoord);

            XMFLOAT3 camera_p[3]{}, raster_p[3]{};
            XMVECTOR ndc_p0 = XMLoadFloat4(&triangle_begin.v0.position);  //clip_space_positions
            XMVECTOR ndc_p1 = XMLoadFloat4(&triangle_begin.v1.position);
            XMVECTOR ndc_p2 = XMLoadFloat4(&triangle_begin.v2.position);

            XMStoreFloat3(&camera_p[0], XMLoadFloat3(&triangle_begin.v0.view_position)); // view_positions
            XMStoreFloat3(&camera_p[1], XMLoadFloat3(&triangle_begin.v1.view_position));
            XMStoreFloat3(&camera_p[2], XMLoadFloat3(&triangle_begin.v2.view_position));

            PerspectiveDivide(camera_p[0], ndc_p0, raster_p[0], imageW, imageH);
            PerspectiveDivide(camera_p[1], ndc_p1, raster_p[1], imageW, imageH);
            PerspectiveDivide(camera_p[2], ndc_p2, raster_p[2], imageW, imageH);

            XMVECTOR v0Raster = XMLoadFloat4(&XMFLOAT4(raster_p[0].x, raster_p[0].y, raster_p[0].z, 1.0f));
            XMVECTOR v1Raster = XMLoadFloat4(&XMFLOAT4(raster_p[1].x, raster_p[1].y, raster_p[1].z, 1.0f));
            XMVECTOR v2Raster = XMLoadFloat4(&XMFLOAT4(raster_p[2].x, raster_p[2].y, raster_p[2].z, 1.0f));

            // compute 1/Z
            XMVECTOR v0RasterZ = _mm_rcp_ps(_mm_permute_ps(v0Raster, _MM_PERM_CCCC))
                , v1RasterZ = _mm_rcp_ps(_mm_permute_ps(v1Raster, _MM_SHUFFLE(2, 2, 2, 2)))
                , v2RasterZ = _mm_rcp_ps(_mm_permute_ps(v2Raster, _MM_SHUFFLE(2, 2, 2, 2)));


            // Precompute reciprocal of vertex z-coordinate
            // Prepare vertex attributes to perspective correction. Divide them by their vertex z-coordinate
            // texture coordinates, vertex positions, vertex normals -  x/z, y/z, z/z, w/z

            tex0 *= v0RasterZ, tex1 *= v1RasterZ, tex2 *= v2RasterZ;
            v0 *= v0RasterZ; v1 *= v1RasterZ; v2 *= v2RasterZ;
            n0 *= v0RasterZ; n1 *= v1RasterZ; n2 *= v2RasterZ;
            // bounding box számítás – ugyanaz:
            XMVECTOR min = _mm_min_ps(v0Raster, _mm_min_ps(v1Raster, v2Raster));
            XMVECTOR max = _mm_max_ps(v0Raster, _mm_max_ps(v1Raster, v2Raster));
            float xmin = XMVectorGetX(min);
            float xmax = XMVectorGetX(max);
            float ymin = XMVectorGetY(min);
            float ymax = XMVectorGetY(max);
            if (xmin > imageW - 1 || xmax < 0 || ymin > imageH - 1 || ymax < 0)
                continue;

            TriWork tw{};
            tw.v0 = v0; tw.v1 = v1; tw.v2 = v2;
            tw.n0 = n0; tw.n1 = n1; tw.n2 = n2;
            tw.tex0 = tex0; tw.tex1 = tex1; tw.tex2 = tex2;
            tw.v0Raster = v0Raster; tw.v1Raster = v1Raster; tw.v2Raster = v2Raster;
            tw.v0RasterZ = v0RasterZ; tw.v1RasterZ = v1RasterZ; tw.v2RasterZ = v2RasterZ;

            tw.bb_xmin = std::max<int32_t>(0, (int32_t)std::floor(xmin));
            tw.bb_xmax = std::min<int32_t>(imageW - 1, (int32_t)std::floor(xmax));
            tw.bb_ymin = std::max<int32_t>(0, (int32_t)std::floor(ymin));
            tw.bb_ymax = std::min<int32_t>(imageH - 1, (int32_t)std::floor(ymax));

            uint32_t triIndex = (uint32_t)triWorkList.size();
            triWorkList.push_back(tw);

            int tx0 = tw.bb_xmin / TILE_SIZE;
            int tx1 = tw.bb_xmax / TILE_SIZE;
            int ty0 = tw.bb_ymin / TILE_SIZE;
            int ty1 = tw.bb_ymax / TILE_SIZE;

            tx0 = std::max<int>(0, tx0);
            ty0 = std::max<int>(0, ty0);
            tx1 = std::min<int>(tilesX - 1, tx1);
            ty1 = std::min<int>(tilesY - 1, ty1);

            for (int ty = ty0; ty <= ty1; ++ty)
                for (int tx = tx0; tx <= tx1; ++tx)
                    bins[ty * tilesX + tx].triIndices.push_back(triIndex);
        }
    }
    // --------------------------- Binning-pass end
    // --------------------------- Raster-pass  begin
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
        workers.emplace_back([&, tid]()
            {
                XMVECTOR light1_pos = XMLoadFloat3(&light1.position);
                XMVECTOR view_pos = XMLoadFloat3(&viewerPosition);
                XMVECTOR Lc = XMLoadFloat4(&light1.color);
                XMVECTOR Ma = XMLoadFloat4(&mesh->mat.ambient);
                XMVECTOR Md = XMLoadFloat4(&mesh->mat.diffuse);
                XMVECTOR Ms = XMLoadFloat4(&mesh->mat.specular);
                float MaterialPower = mesh->mat.power;
                XMVECTOR Ka = Ma * light1.globalAmbient;
                XMVECTOR zero_vector = _mm_setzero_ps();

                for (int tileIndex = tid; tileIndex < tilesX * tilesY; tileIndex += numThreads)
                {
                    int tx = tileIndex % tilesX;
                    int ty = tileIndex / tilesX;

                    uint32_t x0 = tx * TILE_SIZE;
                    uint32_t y0 = ty * TILE_SIZE;
                    uint32_t x1 = std::min<uint32_t>(x0 + TILE_SIZE - 1, imageW - 1);
                    uint32_t y1 = std::min<uint32_t>(y0 + TILE_SIZE - 1, imageH - 1);

                    auto& bin = bins[tileIndex];
                    for (uint32_t triIdx : bin.triIndices)
                    {
                        const TriWork& tw = triWorkList[triIdx];
                        // Tile-corner coverage test for early tile-rejection
                        if (TileOutsideTriangle(tw, x0, y0, x1, y1))
                            continue;
                        // bounding box tile-ra szűkítve
                        uint32_t bb_xmin = std::max<uint32_t>(x0, tw.bb_xmin);
                        uint32_t bb_xmax = std::min<uint32_t>(x1, tw.bb_xmax);
                        uint32_t bb_ymin = std::max<uint32_t>(y0, tw.bb_ymin);
                        uint32_t bb_ymax = std::min<uint32_t>(y1, tw.bb_ymax);
                        if (bb_xmin > bb_xmax || bb_ymin > bb_ymax)
                            continue;

                        XMVECTOR v0 = tw.v0, v1 = tw.v1, v2 = tw.v2;
                        XMVECTOR n0 = tw.n0, n1 = tw.n1, n2 = tw.n2;
                        XMVECTOR tex0 = tw.tex0, tex1 = tw.tex1, tex2 = tw.tex2;
                        XMVECTOR v0Raster = tw.v0Raster, v1Raster = tw.v1Raster, v2Raster = tw.v2Raster;
                        XMVECTOR v0RasterZ = tw.v0RasterZ, v1RasterZ = tw.v1RasterZ, v2RasterZ = tw.v2RasterZ;

                  
                        XMVECTOR zero_vector = _mm_setzero_ps();
                        // calculate 1/area  (the reciprocal of the triangle`s area ) with edge equation
                        //generate vectors to calculate triangle`s area 
                        XMVECTOR v2x_v2y_v2x_v2y = _mm_permute_ps(v2Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
                        XMVECTOR v0x_v0y_v0x_v0y = _mm_permute_ps(v0Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
                        XMVECTOR v0x_v0y_0_0 = _mm_shuffle_ps(v0x_v0y_v0x_v0y, zero_vector, _MM_SHUFFLE(0, 0, 1, 0)); // helper vector = | v0x v0y 0 0 |

                        XMVECTOR v1y_v1x_v1y_v1x = _mm_permute_ps(v1Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA = yxyx
                        XMVECTOR v0y_v0x_v0y_v0x = _mm_permute_ps(v0Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA= yxyx

                        //generate vectors for the coefficients of the edge equation  dY12, dY20, dY01 etc.
                        XMVECTOR v2y_v0y_v2x_v0x = _mm_permute_ps(_mm_unpacklo_ps(v0Raster, v2Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
                        XMVECTOR v1y_v2y_v1x_v2x = _mm_permute_ps(_mm_unpacklo_ps(v2Raster, v1Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
                        //the differences are used in edge equation:  dY12=v2y-v1y , dY20=v0y-v2y, dX12=v2x-v1x, dX20=v0x-v2x
                        XMVECTOR dY12_dY20_dX12_dX20 = v2y_v0y_v2x_v0x - v1y_v2y_v1x_v2x;
                        // dY01 = v1y-v0y , dX01=v1x-v0x
                        XMVECTOR dY01_dX01_dY01_dX01 = v1y_v1x_v1y_v1x - v0y_v0x_v0y_v0x;
                        //dX02 and dY02 are used for area calculation only!
                        XMVECTOR dX02_dY02_dX02_dY02 = v2x_v2y_v2x_v2y - v0x_v0y_v0x_v0y;

                      

                        //calculate the reciprocal of area (v2x-v0x)*(v1y-v0y) - (v2y-v0y)*(v1x-v0x)
                        XMVECTOR dxdy = _mm_mul_ps(dX02_dY02_dX02_dY02, dY01_dX01_dY01_dX01);
                        XMVECTOR rcp_area = _mm_rcp_ps(_mm_hsub_ps(dxdy, dxdy));
                        rcp_area = _mm_insert_ps(rcp_area, zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0));  // reciprocal area vector | xyzw | =  | 1/area  1/area  1/area  0.0 |
                        // --- edge setup (pontosan a mostani kódod) ---
                        

                       

                        XMVECTOR PxPyPz = XMVectorSet(bb_xmin + 0.5f, bb_ymin + 0.5f, 0.0f, 0.0f);
                        // ... (ugyanaz a PxPxPx_0, PyPyPy_0, dY12_dY20_dY01_0, dX12_dX20_dX01_0, w_0_w_1_w_2, stb.)
                        XMVECTOR PxPxPx_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(0, 0, 0, 0)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_AAAA = xxxx
                        XMVECTOR PyPyPy_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(1, 1, 1, 1)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_BBBB = yyyy
                        dY01_dX01_dY01_dX01 = _mm_insert_ps(dY01_dX01_dY01_dX01, zero_vector, _MM_MK_INSERTPS_NDX(0, 1, 4)); // the new vector:  | d01y 0 0 d01x |
                        XMVECTOR dY12_dY20_dY01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(1, 0, 1, 0)); //ABAB
                        XMVECTOR dX12_dX20_dX01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(2, 3, 3, 2));//CDDC
                        XMVECTOR v1x_v2x_v0x_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 0, 3, 2)); // | v1x v2x v0x 0 |
                        XMVECTOR v1y_v2y_v0y_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 1, 1, 0)); //  | v1y v2y v0y 0 |
                        //XMVECTOR w_0_w_1_w_2 = _mm_fnmadd_ps(dX12_dX20_dX01_0, _mm_sub_ps(PyPyPy_0, v1y_v2y_v0y_0), _mm_mul_ps(dY12_dY20_dY01_0, _mm_sub_ps(PxPxPx_0, v1x_v2x_v0x_0)));
                        XMVECTOR w_0_w_1_w_2 = (PxPxPx_0 - v1x_v2x_v0x_0) * dY12_dY20_dY01_0 - (PyPyPy_0 - v1y_v2y_v0y_0) * dX12_dX20_dX01_0;

                        //make vector  of  | v0z v1z v2z 0 | for depth testing
                        XMVECTOR v0z_v1z_v2z_0 = _mm_insert_ps(v0RasterZ, v1RasterZ, _MM_MK_INSERTPS_NDX(0, 1, 12));
                        v0z_v1z_v2z_0 = _mm_insert_ps(v0z_v1z_v2z_0, v2RasterZ, _MM_MK_INSERTPS_NDX(0, 2, 0));

                        for (uint32_t y = bb_ymin; y <= bb_ymax; ++y)
                        {
                            XMVECTOR w0_w1_w2 = w_0_w_1_w_2;

                            for (uint32_t x = bb_xmin; x <= bb_xmax; ++x)
                            {
                                int mask = _mm_movemask_ps(w0_w1_w2);
                                if ((mask & 7) == 0 || (mask & 7) == 7)
                                {
                                    XMVECTOR bary = w0_w1_w2 * rcp_area;

                                    float z = 1.0f / XMVectorGetX(XMVector3Dot(v0z_v1z_v2z_0, bary));
                                    z = clamp(z, nearClippingPLane, farClippingPLane);

                                    float b0 = XMVectorGetX(bary);
                                    float b1 = XMVectorGetY(bary);
                                    float b2 = XMVectorGetZ(bary);

                                    uint64_t idxPix = (uint64_t)y * imageW + x;
                                    if (z < depthBuffer[idxPix])
                                    {
                                        z += 0.001f;
                                        depthBuffer[idxPix] = z;
                                        
                                        // shader – pontosan a mostani kódod:
                                        XMVECTOR texcoord = tex0 * b0 + tex1 * b1 + tex2 * b2;
                                        texcoord *= z;

                                        XMVECTOR pt = v0 * b0 + v1 * b1 + v2 * b2;
                                        pt *= z;

                                        XMVECTOR N = n0 * b0 + n1 * b1 + n2 * b2;
                                        N *= z;
                                        N = XMVector3NormalizeEst(N);

                                        XMVECTOR L = light1_pos - pt;
                                        float distance = XMVectorGetX(XMVector3Length(L));
                                        L /= distance;
                                        float attenuation = 1.0f / (light1.constantAttenuation + light1.linearAttenuation * distance + light1.quadraticAttenuation * distance * distance);

                                        XMVECTOR V = view_pos - pt;
                                        V = XMVector3NormalizeEst(V);
                                        XMVECTOR H = XMVector3NormalizeEst(L + V);

                                        XMVECTOR NdotL = XMVectorMax(XMVector3Dot(N, L), zero_vector);
                                        XMVECTOR NdotH = XMVectorMax(XMVector3Dot(N, H), zero_vector);

                                        XMVECTOR Kd = NdotL * (Lc * Md) * attenuation;
                                        XMVECTOR Ks = pow(XMVectorGetX(NdotH), MaterialPower) * (Lc * Ms) * attenuation;

                                        XMVECTOR finalcolor = XMVectorSaturate((Ka + Kd + Ks));

                                        float checker = (fmod(XMVectorGetX(texcoord) * 6, 1.0f) > 0.2f)
                                            ^ (fmod(XMVectorGetY(texcoord) * 20, 1.0f) < 0.3f);
                                        float pattern = 0.9f * (1.0f - checker) + 1.0f * checker;

                                        finalcolor *= pattern;
                                        finalcolor *= 255.0f;

                                        frameBuffer[idxPix] = finalcolor;
                                    }
                                }

                                w0_w1_w2 += dY12_dY20_dY01_0;
                            }

                            w_0_w_1_w_2 -= dX12_dX20_dX01_0;
                        }
                    }
                }
            });
    }
    // --------------------------- Raster-pass end
    for (auto& t : workers) if (t.joinable()) t.join();
}

//====================================================================
//====================================================================
// SoA friendly code!
// scalar pre-calculated attributes for 1 triangle 
// use instead of struct TriWork !
struct TriPacketSoA
{
    // clip-space raster coords
    float v0x, v0y, v0z;
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;

    // 1/z értékek
    float invZ0, invZ1, invZ2;

    // perspective-correct attribútumok (már előre osztva Z-vel)
    float tex0u, tex0v;
    float tex1u, tex1v;
    float tex2u, tex2v;

    float n0x, n0y, n0z;
    float n1x, n1y, n1z;
    float n2x, n2y, n2z;

    float p0x, p0y, p0z;
    float p1x, p1y, p1z;
    float p2x, p2y, p2z;

    // bounding box
    uint32_t xmin, xmax;
    uint32_t ymin, ymax;
};
//Ez váltja ki a jelenlegi TriWork‑ot, Scalar TriPacketSoA kitöltése MeshSoAFrame‑ből:
TriPacketSoA BuildTriPacket(
    const MeshSoAFrame& mesh,
    uint32_t i0, uint32_t i1, uint32_t i2,
    uint32_t imageW, uint32_t imageH)
{
    TriPacketSoA out{};

    // clip-space → NDC → raster
    auto loadClip = [&](uint32_t idx, float& x, float& y, float& z, float& w)
        {
            x = mesh.clipX[idx];
            y = mesh.clipY[idx];
            z = mesh.clipZ[idx];
            w = mesh.clipW[idx];
        };

    float x0, y0, z0, w0;
    float x1, y1, z1, w1;
    float x2, y2, z2, w2;

    loadClip(i0, x0, y0, z0, w0);
    loadClip(i1, x1, y1, z1, w1);
    loadClip(i2, x2, y2, z2, w2);

    // perspective divide
    float ndc0x = x0 / w0, ndc0y = y0 / w0, ndc0z = z0 / w0;
    float ndc1x = x1 / w1, ndc1y = y1 / w1, ndc1z = z1 / w1;
    float ndc2x = x2 / w2, ndc2y = y2 / w2, ndc2z = z2 / w2;

    // viewport transform
    auto toRaster = [&](float ndcx, float ndcy, float ndcz,
        float& rx, float& ry, float& rz)
        {
            rx = (ndcx * 0.5f + 0.5f) * imageW;
            ry = (1.0f - (ndcy * 0.5f + 0.5f)) * imageH;
            rz = ndcz;
        };

    toRaster(ndc0x, ndc0y, ndc0z, out.v0x, out.v0y, out.v0z);
    toRaster(ndc1x, ndc1y, ndc1z, out.v1x, out.v1y, out.v1z);
    toRaster(ndc2x, ndc2y, ndc2z, out.v2x, out.v2y, out.v2z);

    // 1/z
    out.invZ0 = 1.0f / out.v0z;
    out.invZ1 = 1.0f / out.v1z;
    out.invZ2 = 1.0f / out.v2z;

    // perspective-correct attribútumok előkészítése
    auto prep = [&](uint32_t idx, float invZ,
        float& tx, float& ty,
        float& nx, float& ny, float& nz,
        float& px, float& py, float& pz)
        {
            tx = mesh.uvU[idx] * invZ;
            ty = mesh.uvV[idx] * invZ;

            nx = mesh.nrmX[idx] * invZ;
            ny = mesh.nrmY[idx] * invZ;
            nz = mesh.nrmZ[idx] * invZ;

            px = mesh.worldX[idx] * invZ;
            py = mesh.worldY[idx] * invZ;
            pz = mesh.worldZ[idx] * invZ;
        };

    prep(i0, out.invZ0, out.tex0u, out.tex0v, out.n0x, out.n0y, out.n0z, out.p0x, out.p0y, out.p0z);
    prep(i1, out.invZ1, out.tex1u, out.tex1v, out.n1x, out.n1y, out.n1z, out.p1x, out.p1y, out.p1z);
    prep(i2, out.invZ2, out.tex2u, out.tex2v, out.n2x, out.n2y, out.n2z, out.p2x, out.p2y, out.p2z);

    // bounding box
    float xmin = std::min<float>({ out.v0x,out.v1x,out.v2x });
    float xmax = std::max<float>({ out.v0x,out.v1x,out.v2x });
    float ymin = std::min<float>({ out.v0y,out.v1y,out.v2y });
    float ymax = std::max<float>({ out.v0y,out.v1y,out.v2y });

    out.xmin = std::max<float>(0, (int)std::floor(xmin));
    out.xmax = std::min<float>((int)imageW - 1, (int)std::floor(xmax));
    out.ymin = std::max<float>(0, (int)std::floor(ymin));
    out.ymax = std::min<float>((int)imageH - 1, (int)std::floor(ymax));

    return out;
}

// pre-calculated attributes for 8-wide triangle packet
struct TriPacket8
{
    __m256 v0x, v0y, v0z;
    __m256 v1x, v1y, v1z;
    __m256 v2x, v2y, v2z;

    __m256 invZ0, invZ1, invZ2;

    __m256 tex0u, tex0v;
    __m256 tex1u, tex1v;
    __m256 tex2u, tex2v;

    __m256 n0x, n0y, n0z;
    __m256 n1x, n1y, n1z;
    __m256 n2x, n2y, n2z;

    __m256 p0x, p0y, p0z;
    __m256 p1x, p1y, p1z;
    __m256 p2x, p2y, p2z;

    __m256 xmin, xmax;
    __m256 ymin, ymax;
};
inline __m256 load8(const float* s0, const float* s1, const float* s2, const float* s3,
    const float* s4, const float* s5, const float* s6, const float* s7)
{
    // scalar gather → 8-wide
    return _mm256_set_ps(*s7, *s6, *s5, *s4, *s3, *s2, *s1, *s0);
}
//AVX2 triangle packet builder: 8 scalar TriPacketSoA → 1 TriPacket8
// 8 scalar → 1 AVX packet: 
TriPacket8 BuildPacket8(const TriPacketSoA* tri[8])
{
    TriPacket8 p{};

    // v0
    p.v0x = _mm256_set_ps(tri[7]->v0x, tri[6]->v0x, tri[5]->v0x, tri[4]->v0x,
        tri[3]->v0x, tri[2]->v0x, tri[1]->v0x, tri[0]->v0x);
    p.v0y = _mm256_set_ps(tri[7]->v0y, tri[6]->v0y, tri[5]->v0y, tri[4]->v0y,
        tri[3]->v0y, tri[2]->v0y, tri[1]->v0y, tri[0]->v0y);
    p.v0z = _mm256_set_ps(tri[7]->v0z, tri[6]->v0z, tri[5]->v0z, tri[4]->v0z,
        tri[3]->v0z, tri[2]->v0z, tri[1]->v0z, tri[0]->v0z);

    // v1
    p.v1x = _mm256_set_ps(tri[7]->v1x, tri[6]->v1x, tri[5]->v1x, tri[4]->v1x,
        tri[3]->v1x, tri[2]->v1x, tri[1]->v1x, tri[0]->v1x);
    p.v1y = _mm256_set_ps(tri[7]->v1y, tri[6]->v1y, tri[5]->v1y, tri[4]->v1y,
        tri[3]->v1y, tri[2]->v1y, tri[1]->v1y, tri[0]->v1y);
    p.v1z = _mm256_set_ps(tri[7]->v1z, tri[6]->v1z, tri[5]->v1z, tri[4]->v1z,
        tri[3]->v1z, tri[2]->v1z, tri[1]->v1z, tri[0]->v1z);

    // v2
    p.v2x = _mm256_set_ps(tri[7]->v2x, tri[6]->v2x, tri[5]->v2x, tri[4]->v2x,
        tri[3]->v2x, tri[2]->v2x, tri[1]->v2x, tri[0]->v2x);
    p.v2y = _mm256_set_ps(tri[7]->v2y, tri[6]->v2y, tri[5]->v2y, tri[4]->v2y,
        tri[3]->v2y, tri[2]->v2y, tri[1]->v2y, tri[0]->v2y);
    p.v2z = _mm256_set_ps(tri[7]->v2z, tri[6]->v2z, tri[5]->v2z, tri[4]->v2z,
        tri[3]->v2z, tri[2]->v2z, tri[1]->v2z, tri[0]->v2z);

    // invZ
    p.invZ0 = _mm256_set_ps(tri[7]->invZ0, tri[6]->invZ0, tri[5]->invZ0, tri[4]->invZ0,
        tri[3]->invZ0, tri[2]->invZ0, tri[1]->invZ0, tri[0]->invZ0);
    p.invZ1 = _mm256_set_ps(tri[7]->invZ1, tri[6]->invZ1, tri[5]->invZ1, tri[4]->invZ1,
        tri[3]->invZ1, tri[2]->invZ1, tri[1]->invZ1, tri[0]->invZ1);
    p.invZ2 = _mm256_set_ps(tri[7]->invZ2, tri[6]->invZ2, tri[5]->invZ2, tri[4]->invZ2,
        tri[3]->invZ2, tri[2]->invZ2, tri[1]->invZ2, tri[0]->invZ2);

    // tex
    p.tex0u = _mm256_set_ps(tri[7]->tex0u, tri[6]->tex0u, tri[5]->tex0u, tri[4]->tex0u,
        tri[3]->tex0u, tri[2]->tex0u, tri[1]->tex0u, tri[0]->tex0u);
    p.tex0v = _mm256_set_ps(tri[7]->tex0v, tri[6]->tex0v, tri[5]->tex0v, tri[4]->tex0v,
        tri[3]->tex0v, tri[2]->tex0v, tri[1]->tex0v, tri[0]->tex0v);
    p.tex1u = _mm256_set_ps(tri[7]->tex1u, tri[6]->tex1u, tri[5]->tex1u, tri[4]->tex1u,
        tri[3]->tex1u, tri[2]->tex1u, tri[1]->tex1u, tri[0]->tex1u);
    p.tex1v = _mm256_set_ps(tri[7]->tex1v, tri[6]->tex1v, tri[5]->tex1v, tri[4]->tex1v,
        tri[3]->tex1v, tri[2]->tex1v, tri[1]->tex1v, tri[0]->tex1v);
    p.tex2u = _mm256_set_ps(tri[7]->tex2u, tri[6]->tex2u, tri[5]->tex2u, tri[4]->tex2u,
        tri[3]->tex2u, tri[2]->tex2u, tri[1]->tex2u, tri[0]->tex2u);
    p.tex2v = _mm256_set_ps(tri[7]->tex2v, tri[6]->tex2v, tri[5]->tex2v, tri[4]->tex2v,
        tri[3]->tex2v, tri[2]->tex2v, tri[1]->tex2v, tri[0]->tex2v);

    // normálok
    p.n0x = _mm256_set_ps(tri[7]->n0x, tri[6]->n0x, tri[5]->n0x, tri[4]->n0x,
        tri[3]->n0x, tri[2]->n0x, tri[1]->n0x, tri[0]->n0x);
    p.n0y = _mm256_set_ps(tri[7]->n0y, tri[6]->n0y, tri[5]->n0y, tri[4]->n0y,
        tri[3]->n0y, tri[2]->n0y, tri[1]->n0y, tri[0]->n0y);
    p.n0z = _mm256_set_ps(tri[7]->n0z, tri[6]->n0z, tri[5]->n0z, tri[4]->n0z,
        tri[3]->n0z, tri[2]->n0z, tri[1]->n0z, tri[0]->n0z);

    p.n1x = _mm256_set_ps(tri[7]->n1x, tri[6]->n1x, tri[5]->n1x, tri[4]->n1x,
        tri[3]->n1x, tri[2]->n1x, tri[1]->n1x, tri[0]->n1x);
    p.n1y = _mm256_set_ps(tri[7]->n1y, tri[6]->n1y, tri[5]->n1y, tri[4]->n1y,
        tri[3]->n1y, tri[2]->n1y, tri[1]->n1y, tri[0]->n1y);
    p.n1z = _mm256_set_ps(tri[7]->n1z, tri[6]->n1z, tri[5]->n1z, tri[4]->n1z,
        tri[3]->n1z, tri[2]->n1z, tri[1]->n1z, tri[0]->n1z);

    p.n2x = _mm256_set_ps(tri[7]->n2x, tri[6]->n2x, tri[5]->n2x, tri[4]->n2x,
        tri[3]->n2x, tri[2]->n2x, tri[1]->n2x, tri[0]->n2x);
    p.n2y = _mm256_set_ps(tri[7]->n2y, tri[6]->n2y, tri[5]->n2y, tri[4]->n2y,
        tri[3]->n2y, tri[2]->n2y, tri[1]->n2y, tri[0]->n2y);
    p.n2z = _mm256_set_ps(tri[7]->n2z, tri[6]->n2z, tri[5]->n2z, tri[4]->n2z,
        tri[3]->n2z, tri[2]->n2z, tri[1]->n2z, tri[0]->n2z);

    // pozíciók
    p.p0x = _mm256_set_ps(tri[7]->p0x, tri[6]->p0x, tri[5]->p0x, tri[4]->p0x,
        tri[3]->p0x, tri[2]->p0x, tri[1]->p0x, tri[0]->p0x);
    p.p0y = _mm256_set_ps(tri[7]->p0y, tri[6]->p0y, tri[5]->p0y, tri[4]->p0y,
        tri[3]->p0y, tri[2]->p0y, tri[1]->p0y, tri[0]->p0y);
    p.p0z = _mm256_set_ps(tri[7]->p0z, tri[6]->p0z, tri[5]->p0z, tri[4]->p0z,
        tri[3]->p0z, tri[2]->p0z, tri[1]->p0z, tri[0]->p0z);

    p.p1x = _mm256_set_ps(tri[7]->p1x, tri[6]->p1x, tri[5]->p1x, tri[4]->p1x,
        tri[3]->p1x, tri[2]->p1x, tri[1]->p1x, tri[0]->p1x);
    p.p1y = _mm256_set_ps(tri[7]->p1y, tri[6]->p1y, tri[5]->p1y, tri[4]->p1y,
        tri[3]->p1y, tri[2]->p1y, tri[1]->p1y, tri[0]->p1y);
    p.p1z = _mm256_set_ps(tri[7]->p1z, tri[6]->p1z, tri[5]->p1z, tri[4]->p1z,
        tri[3]->p1z, tri[2]->p1z, tri[1]->p1z, tri[0]->p1z);

    p.p2x = _mm256_set_ps(tri[7]->p2x, tri[6]->p2x, tri[5]->p2x, tri[4]->p2x,
        tri[3]->p2x, tri[2]->p2x, tri[1]->p2x, tri[0]->p2x);
    p.p2y = _mm256_set_ps(tri[7]->p2y, tri[6]->p2y, tri[5]->p2y, tri[4]->p2y,
        tri[3]->p2y, tri[2]->p2y, tri[1]->p2y, tri[0]->p2y);
    p.p2z = _mm256_set_ps(tri[7]->p2z, tri[6]->p2z, tri[5]->p2z, tri[4]->p2z,
        tri[3]->p2z, tri[2]->p2z, tri[1]->p2z, tri[0]->p2z);

    // bounding box
    p.xmin = _mm256_set_ps((float)tri[7]->xmin, (float)tri[6]->xmin, (float)tri[5]->xmin, (float)tri[4]->xmin,
        (float)tri[3]->xmin, (float)tri[2]->xmin, (float)tri[1]->xmin, (float)tri[0]->xmin);
    p.xmax = _mm256_set_ps((float)tri[7]->xmax, (float)tri[6]->xmax, (float)tri[5]->xmax, (float)tri[4]->xmax,
        (float)tri[3]->xmax, (float)tri[2]->xmax, (float)tri[1]->xmax, (float)tri[0]->xmax);
    p.ymin = _mm256_set_ps((float)tri[7]->ymin, (float)tri[6]->ymin, (float)tri[5]->ymin, (float)tri[4]->ymin,
        (float)tri[3]->ymin, (float)tri[2]->ymin, (float)tri[1]->ymin, (float)tri[0]->ymin);
    p.ymax = _mm256_set_ps((float)tri[7]->ymax, (float)tri[6]->ymax, (float)tri[5]->ymax, (float)tri[4]->ymax,
        (float)tri[3]->ymax, (float)tri[2]->ymax, (float)tri[1]->ymax, (float)tri[0]->ymax);

    return p;
}


void Render::DrawMeshTiledBinningSoA(const MeshSoAFrame& mesh, const Material& mat)
{
    const uint32_t imageW = ImageWidth;
    const uint32_t imageH = ImageHeight;

    const int TILE_SIZE = 32;
    const int tilesX = (imageW + TILE_SIZE - 1) / TILE_SIZE;
    const int tilesY = (imageH + TILE_SIZE - 1) / TILE_SIZE;

    struct TileBin
    {
        std::vector<uint32_t> triIndices; // index TriWork-ba
    };

    std::vector<TriWork> triWorkList;
    triWorkList.reserve(mesh.IndexCount() / 3); // clipping miatt lehet több

    std::vector<TileBin> bins(tilesX * tilesY);

    // --------------------------- Binning-pass begin
    for (uint32_t i = 0; i < mesh.IndexCount() / 3; i++)
    {
        std::vector<Vertex> clipped_vertices(3); // input to clipTriangle()
        std::vector<Triangle> clipped_triangles{}; // output from clipTriangle()

        // read triangle data from SoA vertex storage
        uint32_t i0 = mesh.indices[i * 3 + 0];
        uint32_t i1 = mesh.indices[i * 3 + 1];
        uint32_t i2 = mesh.indices[i * 3 + 2];
        // clip-space
        clipped_vertices[0].position = float4{ mesh.clipX[i0], mesh.clipY[i0], mesh.clipZ[i0], mesh.clipW[i0] };
        clipped_vertices[1].position = float4{ mesh.clipX[i1], mesh.clipY[i1], mesh.clipZ[i1], mesh.clipW[i1] };
        clipped_vertices[2].position = float4{ mesh.clipX[i2], mesh.clipY[i2], mesh.clipZ[i2], mesh.clipW[i2] };

        // view-space
        clipped_vertices[0].view_position = float3{ mesh.viewX[i0], mesh.viewY[i0], mesh.viewZ[i0] };
        clipped_vertices[1].view_position = float3{ mesh.viewX[i1], mesh.viewY[i1], mesh.viewZ[i1] };
        clipped_vertices[2].view_position = float3{ mesh.viewX[i2], mesh.viewY[i2], mesh.viewZ[i2] };

        // world-space
        clipped_vertices[0].world_position = float4{ mesh.worldX[i0], mesh.worldY[i0], mesh.worldZ[i0], 1.0f };
        clipped_vertices[1].world_position = float4{ mesh.worldX[i1], mesh.worldY[i1], mesh.worldZ[i1], 1.0f };
        clipped_vertices[2].world_position = float4{ mesh.worldX[i2], mesh.worldY[i2], mesh.worldZ[i2], 1.0f };

        // normals
        clipped_vertices[0].normal = float4{ mesh.nrmX[i0], mesh.nrmY[i0], mesh.nrmZ[i0], 0.0f };
        clipped_vertices[1].normal = float4{ mesh.nrmX[i1], mesh.nrmY[i1], mesh.nrmZ[i1], 0.0f };
        clipped_vertices[2].normal = float4{ mesh.nrmX[i2], mesh.nrmY[i2], mesh.nrmZ[i2], 0.0f };

        // texcoord
        clipped_vertices[0].texcoord = float2{ mesh.uvU[i0], mesh.uvV[i0] };
        clipped_vertices[1].texcoord = float2{ mesh.uvU[i1], mesh.uvV[i1] };
        clipped_vertices[2].texcoord = float2{ mesh.uvU[i2], mesh.uvV[i2] };

        std::array<float, 4> plane = { 0, 0, 1, -1 };  // clipping triangles along Z-axis

        clipped_triangles = clipTriangle(Triangle(clipped_vertices[0], clipped_vertices[1], clipped_vertices[2]), plane); // use depth clipping only

        for (auto& triangle_begin : clipped_triangles)
        {
            // v0,v1,v2, n0,n1,n2, tex0,tex1,tex2, camera_p, raster_p, v0Raster,v1Raster,v2Raster,
            // v0RasterZ,v1RasterZ,v2RasterZ – pontosan ugyanúgy, mint most
            XMVECTOR v0 = XMLoadFloat4(&triangle_begin.v0.world_position);
            XMVECTOR v1 = XMLoadFloat4(&triangle_begin.v1.world_position);
            XMVECTOR v2 = XMLoadFloat4(&triangle_begin.v2.world_position);

            XMVECTOR n0 = XMLoadFloat4(&triangle_begin.v0.normal);
            XMVECTOR n1 = XMLoadFloat4(&triangle_begin.v1.normal);
            XMVECTOR n2 = XMLoadFloat4(&triangle_begin.v2.normal);

            XMVECTOR tex0 = XMLoadFloat2(&triangle_begin.v0.texcoord);
            XMVECTOR tex1 = XMLoadFloat2(&triangle_begin.v1.texcoord);
            XMVECTOR tex2 = XMLoadFloat2(&triangle_begin.v2.texcoord);

            XMFLOAT3 camera_p[3]{}, raster_p[3]{};
            XMVECTOR ndc_p0 = XMLoadFloat4(&triangle_begin.v0.position);  //clip_space_positions
            XMVECTOR ndc_p1 = XMLoadFloat4(&triangle_begin.v1.position);
            XMVECTOR ndc_p2 = XMLoadFloat4(&triangle_begin.v2.position);

            XMStoreFloat3(&camera_p[0], XMLoadFloat3(&triangle_begin.v0.view_position)); // view_positions
            XMStoreFloat3(&camera_p[1], XMLoadFloat3(&triangle_begin.v1.view_position));
            XMStoreFloat3(&camera_p[2], XMLoadFloat3(&triangle_begin.v2.view_position));

            PerspectiveDivide(camera_p[0], ndc_p0, raster_p[0], imageW, imageH);
            PerspectiveDivide(camera_p[1], ndc_p1, raster_p[1], imageW, imageH);
            PerspectiveDivide(camera_p[2], ndc_p2, raster_p[2], imageW, imageH);

            XMVECTOR v0Raster = XMLoadFloat4(&XMFLOAT4(raster_p[0].x, raster_p[0].y, raster_p[0].z, 1.0f));
            XMVECTOR v1Raster = XMLoadFloat4(&XMFLOAT4(raster_p[1].x, raster_p[1].y, raster_p[1].z, 1.0f));
            XMVECTOR v2Raster = XMLoadFloat4(&XMFLOAT4(raster_p[2].x, raster_p[2].y, raster_p[2].z, 1.0f));

            // compute 1/Z
            XMVECTOR v0RasterZ = _mm_rcp_ps(_mm_permute_ps(v0Raster, _MM_PERM_CCCC))
                , v1RasterZ = _mm_rcp_ps(_mm_permute_ps(v1Raster, _MM_SHUFFLE(2, 2, 2, 2)))
                , v2RasterZ = _mm_rcp_ps(_mm_permute_ps(v2Raster, _MM_SHUFFLE(2, 2, 2, 2)));


            // Precompute reciprocal of vertex z-coordinate
            // Prepare vertex attributes to perspective correction. Divide them by their vertex z-coordinate
            // texture coordinates, vertex positions, vertex normals -  x/z, y/z, z/z, w/z

            tex0 *= v0RasterZ, tex1 *= v1RasterZ, tex2 *= v2RasterZ;
            v0 *= v0RasterZ; v1 *= v1RasterZ; v2 *= v2RasterZ;
            n0 *= v0RasterZ; n1 *= v1RasterZ; n2 *= v2RasterZ;
            // bounding box számítás – ugyanaz:
            XMVECTOR min = _mm_min_ps(v0Raster, _mm_min_ps(v1Raster, v2Raster));
            XMVECTOR max = _mm_max_ps(v0Raster, _mm_max_ps(v1Raster, v2Raster));
            float xmin = XMVectorGetX(min);
            float xmax = XMVectorGetX(max);
            float ymin = XMVectorGetY(min);
            float ymax = XMVectorGetY(max);
            if (xmin > imageW - 1 || xmax < 0 || ymin > imageH - 1 || ymax < 0)
                continue;

            TriWork tw{};
            tw.v0 = v0; tw.v1 = v1; tw.v2 = v2;
            tw.n0 = n0; tw.n1 = n1; tw.n2 = n2;
            tw.tex0 = tex0; tw.tex1 = tex1; tw.tex2 = tex2;
            tw.v0Raster = v0Raster; tw.v1Raster = v1Raster; tw.v2Raster = v2Raster;
            tw.v0RasterZ = v0RasterZ; tw.v1RasterZ = v1RasterZ; tw.v2RasterZ = v2RasterZ;

            tw.bb_xmin = std::max<int32_t>(0, (int32_t)std::floor(xmin));
            tw.bb_xmax = std::min<int32_t>(imageW - 1, (int32_t)std::floor(xmax));
            tw.bb_ymin = std::max<int32_t>(0, (int32_t)std::floor(ymin));
            tw.bb_ymax = std::min<int32_t>(imageH - 1, (int32_t)std::floor(ymax));

            uint32_t triIndex = (uint32_t)triWorkList.size();
            triWorkList.push_back(tw);

            int tx0 = tw.bb_xmin / TILE_SIZE;
            int tx1 = tw.bb_xmax / TILE_SIZE;
            int ty0 = tw.bb_ymin / TILE_SIZE;
            int ty1 = tw.bb_ymax / TILE_SIZE;

            tx0 = std::max<int>(0, tx0);
            ty0 = std::max<int>(0, ty0);
            tx1 = std::min<int>(tilesX - 1, tx1);
            ty1 = std::min<int>(tilesY - 1, ty1);

            for (int ty = ty0; ty <= ty1; ++ty)
                for (int tx = tx0; tx <= tx1; ++tx)
                    bins[ty * tilesX + tx].triIndices.push_back(triIndex);
        }
    }
    // --------------------------- Binning-pass end
    // --------------------------- Raster-pass  begin
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
        workers.emplace_back([&, tid]()
            {
                XMVECTOR light1_pos = XMLoadFloat3(&light1.position);
                XMVECTOR view_pos = XMLoadFloat3(&viewerPosition);
                XMVECTOR Lc = XMLoadFloat4(&light1.color);
                XMVECTOR Ma = XMLoadFloat4(&mat.ambient);
                XMVECTOR Md = XMLoadFloat4(&mat.diffuse);
                XMVECTOR Ms = XMLoadFloat4(&mat.specular);
                float MaterialPower = mat.power;
                XMVECTOR Ka = Ma * light1.globalAmbient;
                XMVECTOR zero_vector = _mm_setzero_ps();

                for (int tileIndex = tid; tileIndex < tilesX * tilesY; tileIndex += numThreads)
                {
                    int tx = tileIndex % tilesX;
                    int ty = tileIndex / tilesX;

                    uint32_t x0 = tx * TILE_SIZE; // tile-corners ( x0, y0, x1, y1)
                    uint32_t y0 = ty * TILE_SIZE;
                    uint32_t x1 = std::min<uint32_t>(x0 + TILE_SIZE - 1, imageW - 1);
                    uint32_t y1 = std::min<uint32_t>(y0 + TILE_SIZE - 1, imageH - 1);

                    auto& bin = bins[tileIndex];
                    for (uint32_t triIdx : bin.triIndices)
                    {
                        const TriWork& tw = triWorkList[triIdx];
                        // Tile-corner coverage test for early tile-rejection
                        if (TileOutsideTriangle(tw, x0, y0, x1, y1))
                            continue;

                        // bounding box tile-ra szűkítve
                        uint32_t bb_xmin = std::max<uint32_t>(x0, tw.bb_xmin);
                        uint32_t bb_xmax = std::min<uint32_t>(x1, tw.bb_xmax);
                        uint32_t bb_ymin = std::max<uint32_t>(y0, tw.bb_ymin);
                        uint32_t bb_ymax = std::min<uint32_t>(y1, tw.bb_ymax);
                        if (bb_xmin > bb_xmax || bb_ymin > bb_ymax)
                            continue;

                        XMVECTOR v0 = tw.v0, v1 = tw.v1, v2 = tw.v2;
                        XMVECTOR n0 = tw.n0, n1 = tw.n1, n2 = tw.n2;
                        XMVECTOR tex0 = tw.tex0, tex1 = tw.tex1, tex2 = tw.tex2;
                        XMVECTOR v0Raster = tw.v0Raster, v1Raster = tw.v1Raster, v2Raster = tw.v2Raster;
                        XMVECTOR v0RasterZ = tw.v0RasterZ, v1RasterZ = tw.v1RasterZ, v2RasterZ = tw.v2RasterZ;


                        XMVECTOR zero_vector = _mm_setzero_ps();
                        // calculate 1/area  (the reciprocal of the triangle`s area ) with edge equation
                        //generate vectors to calculate triangle`s area 
                        XMVECTOR v2x_v2y_v2x_v2y = _mm_permute_ps(v2Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
                        XMVECTOR v0x_v0y_v0x_v0y = _mm_permute_ps(v0Raster, _MM_SHUFFLE(1, 0, 1, 0));//_MM_PERM_ABAB  = xyxy
                        XMVECTOR v0x_v0y_0_0 = _mm_shuffle_ps(v0x_v0y_v0x_v0y, zero_vector, _MM_SHUFFLE(0, 0, 1, 0)); // helper vector = | v0x v0y 0 0 |

                        XMVECTOR v1y_v1x_v1y_v1x = _mm_permute_ps(v1Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA = yxyx
                        XMVECTOR v0y_v0x_v0y_v0x = _mm_permute_ps(v0Raster, _MM_SHUFFLE(0, 1, 0, 1));//_MM_PERM_BABA= yxyx

                        //generate vectors for the coefficients of the edge equation  dY12, dY20, dY01 etc.
                        XMVECTOR v2y_v0y_v2x_v0x = _mm_permute_ps(_mm_unpacklo_ps(v0Raster, v2Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
                        XMVECTOR v1y_v2y_v1x_v2x = _mm_permute_ps(_mm_unpacklo_ps(v2Raster, v1Raster), _MM_SHUFFLE(0, 1, 2, 3)); //_MM_PERM_DCBA
                        //the differences are used in edge equation:  dY12=v2y-v1y , dY20=v0y-v2y, dX12=v2x-v1x, dX20=v0x-v2x
                        XMVECTOR dY12_dY20_dX12_dX20 = v2y_v0y_v2x_v0x - v1y_v2y_v1x_v2x;
                        // dY01 = v1y-v0y , dX01=v1x-v0x
                        XMVECTOR dY01_dX01_dY01_dX01 = v1y_v1x_v1y_v1x - v0y_v0x_v0y_v0x;
                        //dX02 and dY02 are used for area calculation only!
                        XMVECTOR dX02_dY02_dX02_dY02 = v2x_v2y_v2x_v2y - v0x_v0y_v0x_v0y;



                        //calculate the reciprocal of area (v2x-v0x)*(v1y-v0y) - (v2y-v0y)*(v1x-v0x)
                        XMVECTOR dxdy = _mm_mul_ps(dX02_dY02_dX02_dY02, dY01_dX01_dY01_dX01);
                        XMVECTOR rcp_area = _mm_rcp_ps(_mm_hsub_ps(dxdy, dxdy));
                        rcp_area = _mm_insert_ps(rcp_area, zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0));  // reciprocal area vector | xyzw | =  | 1/area  1/area  1/area  0.0 |
                        // --- edge setup (pontosan a mostani kódod) ---




                        XMVECTOR PxPyPz = XMVectorSet(bb_xmin + 0.5f, bb_ymin + 0.5f, 0.0f, 0.0f);
                        // ... (ugyanaz a PxPxPx_0, PyPyPy_0, dY12_dY20_dY01_0, dX12_dX20_dX01_0, w_0_w_1_w_2, stb.)
                        XMVECTOR PxPxPx_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(0, 0, 0, 0)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_AAAA = xxxx
                        XMVECTOR PyPyPy_0 = _mm_insert_ps(_mm_permute_ps(PxPyPz, _MM_SHUFFLE(1, 1, 1, 1)), zero_vector, _MM_MK_INSERTPS_NDX(0, 3, 0)); //_MM_PERM_BBBB = yyyy
                        dY01_dX01_dY01_dX01 = _mm_insert_ps(dY01_dX01_dY01_dX01, zero_vector, _MM_MK_INSERTPS_NDX(0, 1, 4)); // the new vector:  | d01y 0 0 d01x |
                        XMVECTOR dY12_dY20_dY01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(1, 0, 1, 0)); //ABAB
                        XMVECTOR dX12_dX20_dX01_0 = _mm_shuffle_ps(dY12_dY20_dX12_dX20, dY01_dX01_dY01_dX01, _MM_SHUFFLE(2, 3, 3, 2));//CDDC
                        XMVECTOR v1x_v2x_v0x_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 0, 3, 2)); // | v1x v2x v0x 0 |
                        XMVECTOR v1y_v2y_v0y_0 = _mm_shuffle_ps(v1y_v2y_v1x_v2x, v0x_v0y_0_0, _MM_SHUFFLE(2, 1, 1, 0)); //  | v1y v2y v0y 0 |
                        //XMVECTOR w_0_w_1_w_2 = _mm_fnmadd_ps(dX12_dX20_dX01_0, _mm_sub_ps(PyPyPy_0, v1y_v2y_v0y_0), _mm_mul_ps(dY12_dY20_dY01_0, _mm_sub_ps(PxPxPx_0, v1x_v2x_v0x_0)));
                        XMVECTOR w_0_w_1_w_2 = (PxPxPx_0 - v1x_v2x_v0x_0) * dY12_dY20_dY01_0 - (PyPyPy_0 - v1y_v2y_v0y_0) * dX12_dX20_dX01_0;

                        //make vector  of  | v0z v1z v2z 0 | for depth testing
                        XMVECTOR v0z_v1z_v2z_0 = _mm_insert_ps(v0RasterZ, v1RasterZ, _MM_MK_INSERTPS_NDX(0, 1, 12));
                        v0z_v1z_v2z_0 = _mm_insert_ps(v0z_v1z_v2z_0, v2RasterZ, _MM_MK_INSERTPS_NDX(0, 2, 0));

                        for (uint32_t y = bb_ymin; y <= bb_ymax; ++y)
                        {
                            XMVECTOR w0_w1_w2 = w_0_w_1_w_2;

                            for (uint32_t x = bb_xmin; x <= bb_xmax; ++x)
                            {
                                int mask = _mm_movemask_ps(w0_w1_w2);
                                if ((mask & 7) == 0 || (mask & 7) == 7)
                                {
                                    XMVECTOR bary = w0_w1_w2 * rcp_area;

                                    float z = 1.0f / XMVectorGetX(XMVector3Dot(v0z_v1z_v2z_0, bary));
                                    z = clamp(z, nearClippingPLane, farClippingPLane);

                                    float b0 = XMVectorGetX(bary);
                                    float b1 = XMVectorGetY(bary);
                                    float b2 = XMVectorGetZ(bary);

                                    uint64_t idxPix = (uint64_t)y * imageW + x;
                                    if (z < depthBuffer[idxPix])
                                    {
                                        z += 0.001f;
                                        depthBuffer[idxPix] = z;

                                        // shader – pontosan a mostani kódod:
                                        XMVECTOR texcoord = tex0 * b0 + tex1 * b1 + tex2 * b2;
                                        texcoord *= z;

                                        XMVECTOR pt = v0 * b0 + v1 * b1 + v2 * b2;
                                        pt *= z;

                                        XMVECTOR N = n0 * b0 + n1 * b1 + n2 * b2;
                                        N *= z;
                                        N = XMVector3NormalizeEst(N);

                                        XMVECTOR L = light1_pos - pt;
                                        float distance = XMVectorGetX(XMVector3Length(L));
                                        L /= distance;
                                        float attenuation = 1.0f / (light1.constantAttenuation + light1.linearAttenuation * distance + light1.quadraticAttenuation * distance * distance);

                                        XMVECTOR V = view_pos - pt;
                                        V = XMVector3NormalizeEst(V);
                                        XMVECTOR H = XMVector3NormalizeEst(L + V);

                                        XMVECTOR NdotL = XMVectorMax(XMVector3Dot(N, L), zero_vector);
                                        XMVECTOR NdotH = XMVectorMax(XMVector3Dot(N, H), zero_vector);

                                        XMVECTOR Kd = NdotL * (Lc * Md) * attenuation;
                                        XMVECTOR Ks = pow(XMVectorGetX(NdotH), MaterialPower) * (Lc * Ms) * attenuation;

                                        XMVECTOR finalcolor = XMVectorSaturate((Ka + Kd + Ks));

                                        float checker = (fmod(XMVectorGetX(texcoord) * 6, 1.0f) > 0.2f)
                                            ^ (fmod(XMVectorGetY(texcoord) * 20, 1.0f) < 0.3f);
                                        float pattern = 0.9f * (1.0f - checker) + 1.0f * checker;
                                        finalcolor *= pattern;
                                        /*
                                          float u = XMVectorGetX(texcoord);
                                          float v = XMVectorGetY(texcoord);

                                           int cellX = (int)floorf(u * 6.0f);
                                           int cellY = (int)floorf(v * 20.0f);
                                           int parity = (cellX ^ cellY) & 1;

                                           float pattern = 0.9f + 0.1f * parity;

                                           finalcolor *= pattern;
                                        */
                                        finalcolor *= 255.0f;

                                        frameBuffer[idxPix] = finalcolor;
                                    }
                                }

                                w0_w1_w2 += dY12_dY20_dY01_0;
                            }

                            w_0_w_1_w_2 -= dX12_dX20_dX01_0;
                        }
                    }
                }
            });
    }
    // --------------------------- Raster-pass end
    for (auto& t : workers) if (t.joinable()) t.join();
}

/// <summary>
/// Right-handed coordinate system
/// eye = camera`s position, 
/// center = target point (the direction where the camera is aimed at ), 
/// up = world`s up direction ( always = {0,1,0} )
/// </summary>
/// <param name="eye"></param>
/// <param name="center"></param>
/// <param name="up"></param>
/// <returns></returns>
float4x4 Camera::MatrixLookAtRH(float3 const& eye, float3 const& center, float3 const& updirection)
{
    float4x4 Result=float4x4::identity();
    float3  look(center - eye); // the direction the camera is looking at will become the z-coordinate of the camera`s view-system (view direction = lookat (center) - eye position)
    look = normalize(look);
    float3 right(cross(look, updirection));// the x-coordinate of camera => look cross up
    right = normalize(right);
    float3 up(cross(right, look));// the y-coordinate of camera
    Result.m11 = right.x;
    Result.m21 = right.y;
    Result.m31 = right.z;
    Result.m41 = -dot(right, eye);
    Result.m12 = up.x;
    Result.m22 = up.y;
    Result.m32 = up.z;
    Result.m42 = -dot(up, eye);
    Result.m13 = -look.x;
    Result.m23 = -look.y;
    Result.m33 = -look.z;
    Result.m43 = dot(look, eye);
    return Result;
}

/// <summary>
/// Left-handed coordinate system camera matrix
/// </summary>
/// <param name="eye"></param>
/// <param name="center"></param>
/// <param name="up"></param>
/// <returns></returns>
float4x4 Camera::MatrixLookAtLH(float3 const& eye, float3 const& center, float3 const& updirection)
{
    float4x4 Result=float4x4::identity();
    float3  look(center - eye); // the direction the camera is looking will become the z-coordinate of the camera
    look = normalize(look);
    float3 right(cross(updirection, look));// the x-coordinate of camera
    right = normalize(right);
    float3 up(cross(look, right));// the y-coordinate of camera
    Result.m11 = right.x;
    Result.m21 = right.y;
    Result.m31 = right.z;
    Result.m41 = -dot(right, eye);
    Result.m12 = up.x;
    Result.m22 = up.y;
    Result.m32 = up.z;
    Result.m42 = -dot(up, eye);
    Result.m13 = look.x;
    Result.m23 = look.y;
    Result.m33 = look.z;
    Result.m43 = -dot(look, eye);

    return Result;
}
// right-handed perspective projection matrix
// convert screen-space to NDC space and z to range of [-1, 1]
float4x4 Camera::PerspectiveFovRH(const float& Fov, const float& aspectRatio, const float& zNear, const float& zFar)
{
    float4x4 Result{};
    float t, b, l, r;
    // top, bottom, left, right image coordinates
    t = tan((Fov / 2) * XM_PI / 180) * zNear;
    b = -t;
    r = t * aspectRatio;
    l = -r;
    //first row
    Result.m11 = 2 * zNear / (r - l);
    Result.m12 = 0.0f;
    Result.m13 = 0.0f;
    Result.m14 = 0.0f;
    //second row
    Result.m21 = 0.0f;
    Result.m22 = 2 * zNear / (t - b);
    Result.m23 = 0.0f;
    Result.m24 = 0.0f;
    //third row
    Result.m31 = (r + l) / (r - l);
    Result.m32 = (t + b) / (t - b);
    Result.m33 = -(zFar + zNear) / (zFar - zNear);  // -zFar / (zFar - zNear)  -> range [0,1]
    Result.m34 = -1.0f;
    //fourth row
    Result.m41 = 0.0f;
    Result.m42 = 0.0f;
    Result.m43 = -2 * zFar * zNear / (zFar - zNear); // - zFar * zNear / (zFar - zNear) -> range [0,1]
    Result.m44 = 0.0f;
    return Result;
}
// left-handed perspective projection matrix
float4x4 Camera::PerspectiveFovLH(const float& Fov, const float& aspectRatio, const float& zNear, const float& zFar)
{
    float4x4 Result{};
    float t, b, l, r;
    // top, bottom, left, right image coordinates
    t = tan((Fov / 2) * XM_PI / 180) * zNear;
    b = -t;
    r = t * aspectRatio;
    l = -r;
    //first row
    Result.m11 = 2 * zNear / (r - l);
    Result.m12 = 0.0f;
    Result.m13 = 0.0f;
    Result.m14 = 0.0f;
    //second row
    Result.m21 = 0.0f;
    Result.m22 = 2 * zNear / (t - b);
    Result.m23 = 0.0f;
    Result.m24 = 0.0f;
    //third row
    Result.m31 = (r + l) / (r - l);
    Result.m32 = (t + b) / (t - b);
    Result.m33 = (zFar + zNear) / (zFar - zNear);
    Result.m34 = 1.0f;
    //fourth row
    Result.m41 = 0.0f;
    Result.m42 = 0.0f;
    Result.m43 = -2 * zFar * zNear / (zFar - zNear);
    Result.m44 = 0.0f;
    return Result;
}
