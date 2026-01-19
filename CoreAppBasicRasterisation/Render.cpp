#include "pch.h"
#include "Render.h"


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

    // compute screen coordinates... top, bottom, left, right
   /* computeScreenCoordinates(
        filmApertureWidth, filmApertureHeight,
        imageWidth, imageHeight,
        FitResolutionGate::kOverscan,
        nearClippingPLane,
        focalLength,
        t, b, l, r);*/
    
   // resize framebuffer and depth-buffer
    depthBuffer.assign((size_t)ImageWidth * ImageHeight, farClippingPLane);
    frameBuffer.assign((size_t)ImageWidth * ImageHeight, { 0,128,255,0 });

    auto t_start = std::chrono::high_resolution_clock::now(); 
 
    for (Mesh mesh : meshes)
    {
        TransformVertices(&mesh);
        DrawMesh(&mesh);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    auto passedTime = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    *time_passed = passedTime;
    //p_bgra is a public pointer to rendered image  
    p_bgra =frameBuffer.data();
}

void Render::TransformVertices(Mesh* mesh)
{
    XMMATRIX World = XMLoadFloat4x4(&mesh->World);
    XMMATRIX InverseTransposeWorld = XMMatrixTranspose(XMMatrixInverse(nullptr, World));
    XMStoreFloat4x4(&mesh->InverseTransposeWorld, InverseTransposeWorld);

    for (auto& n : mesh->normals)
    {
        float3 invn;
        XMStoreFloat3(&invn, XMVector3TransformNormal(XMLoadFloat3(&n), InverseTransposeWorld));
        mesh->world_normals.push_back(float4(invn, 0.0f));
    }

    for (auto& p : mesh->positions)
    {
        XMVECTOR v = XMVector3Transform(XMLoadFloat3(&p), World);
        float3 wp{};
        XMStoreFloat3(&wp, v);
        mesh->world_positions.push_back(float4(wp, 1.0f));
       
        XMFLOAT3 camera_p; XMFLOAT4 ndc_p;
        Project(v, camera_p, ndc_p);
        mesh->clip_space_positions.push_back(float4(ndc_p.x, ndc_p.y, ndc_p.z, ndc_p.w));
        mesh->view_positions.push_back(float3(camera_p.x, camera_p.y, camera_p.z));
    }    
 
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> 
//The Sutherland - Hodgman algorithm is traditionally used for 2D polygon clipping, but it can be extended to 3D for clipping triangles against a 3D clipping volume(e.g., a frustum).
// use homogeneous coordinates for positions
// Function to check if a point is inside the clipping plane
static bool isInside(const Vertex & point, const std::array<float, 4>&plane)
{
    return plane[0] * point.position.x + plane[1] * point.position.y + plane[2] * point.position.z + plane[3] >= 0;
}

// Function to compute the intersection of a line segment with a clipping plane
static Vertex intersect(const Vertex & p1, const Vertex & p2, const std::array<float, 4>&plane)
{
    Vertex v{};
    
    float d1 = plane[0] * p1.position.x + plane[1] * p1.position.y + plane[2] * p1.position.z + plane[3];  // dot(p1.position, plane)
    float d2 = plane[0] * p2.position.x + plane[1] * p2.position.y + plane[2] * p2.position.z + plane[3]; // dot(p2.position, plane)
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
void Render::computeScreenCoordinates(const float& filmApertureWidth, const float& filmApertureHeight, 
    const uint32_t& imageWidth, const uint32_t& imageHeight, const FitResolutionGate& fitFilm, const float& nearClippingPLane, 
    const float& focalLength, float& top, float& bottom, float& left, float& right)
{
    float filmAspectRatio = filmApertureWidth / filmApertureHeight;
    float deviceAspectRatio = imageWidth / (float)imageHeight;

    top = ((filmApertureHeight * inchToMm / 2) / focalLength) * nearClippingPLane;
    right = ((filmApertureWidth * inchToMm / 2) / focalLength) * nearClippingPLane;

    // field of view (horizontal)
    float fov = 2 * 180 / XM_PI * atan((filmApertureWidth * inchToMm / 2) / focalLength);

    float xscale = 1;
    float yscale = 1;

    switch (fitFilm) {
    default:
    case FitResolutionGate::kFill:
        if (filmAspectRatio > deviceAspectRatio) {
            xscale = deviceAspectRatio / filmAspectRatio;
        }
        else {
            yscale = filmAspectRatio / deviceAspectRatio;
        }
        break;
    case FitResolutionGate::kOverscan:
        if (filmAspectRatio > deviceAspectRatio) {
            yscale = filmAspectRatio / deviceAspectRatio;
        }
        else {
            xscale = deviceAspectRatio / filmAspectRatio;
        }
        break;
    }
    right *= xscale;
    top *= yscale;
    bottom = -top;
    left = -right;
}

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
    XMStoreFloat4(&vertexNDC ,XMVector4Transform(XMLoadFloat3(&vertexCamera), XMLoadFloat4x4(&CameraToScreen))); // Perspective Projection transform
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



void Mesh::ComputeVertexNormals(float3* vertices, uint32_t* Indices, uint32_t numvertices, uint32_t numtriangles, float3* vertexnormals)
{
    vector<VertexIndex> sameIndices;
    vector<float3> facenormals;
    for (uint32_t index = 0; index < numvertices; index++)
    {
        for (uint32_t triangle = 0; triangle < numtriangles; triangle++)
        {
            uint32_t index0 = Indices[triangle * 3];
            uint32_t index1 = Indices[triangle * 3 + 1];
            uint32_t index2 = Indices[triangle * 3 + 2];

            if ((index == index0) || (index == index1) || (index == index2))
            {
               // uint32_t  vi[3]{ index0,index1,index2 };
                VertexIndex vi{ index0,index1,index2 };                
                sameIndices.push_back(vi);
            }
        }

        for (VertexIndex i : sameIndices)
        {
            float3 v0 = vertices[i.v0];
            float3 v1 = vertices[i.v1];
            float3 v2 = vertices[i.v2];
            float3 N = cross((v1 - v0), (v2 - v0));
            N = normalize(N);
            facenormals.push_back(N);
        }
        float3 vertex_normal;
        for (float3 n : facenormals)
        {
            vertex_normal = vertex_normal + n;
        }
        vertex_normal = normalize(vertex_normal);
        vertexnormals[index] = vertex_normal;
        sameIndices.erase(sameIndices.begin(), sameIndices.end());
        facenormals.erase(facenormals.begin(), facenormals.end());
    }
}

// Create a shere with the parametric equation
void Mesh::CreateSphere(float radius, int slices, int stacks)
{
    for (int i = 0; i <= stacks; ++i)
    {
        // V texture coordinate.
        float V = i / (float)stacks;
        float phi = V * XM_PI;

        for (int j = 0; j <= slices; ++j)
        {
            // U texture coordinate.
            float U = j / (float)slices;
            float theta = U * XM_2PI;

            float X = cos(theta) * sin(phi);
            float Y = cos(phi);
            float Z = sin(theta) * sin(phi);

            positions.push_back(float3(X, Y, Z) * radius);
            normals.push_back(float3(X, Y, Z));
            textureCoords.push_back(float2(U, V));
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
    texcoordIndices = indices;
    number_of_triangles = numtri;
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
        std::vector<Vertex> clipped_vertices{};
        clipped_vertices.push_back(Vertex());
        clipped_vertices.push_back(Vertex());
        clipped_vertices.push_back(Vertex());
        std::vector<Triangle> clipped_triangles{};
        
        
        // read triangle data
        //read clip-space position
        clipped_vertices[0].position = mesh->clip_space_positions[mesh->indices[i * 3]];
        clipped_vertices[1].position = mesh->clip_space_positions[mesh->indices[i * 3 + 1]];
        clipped_vertices[2].position = mesh->clip_space_positions[mesh->indices[i * 3 + 2]];
        //read view-space position
        clipped_vertices[0].view_position = mesh->view_positions[mesh->indices[i * 3]];
        clipped_vertices[1].view_position = mesh->view_positions[mesh->indices[i * 3 + 1]];
        clipped_vertices[2].view_position = mesh->view_positions[mesh->indices[i * 3 + 2]];
        //read world-space position
        clipped_vertices[0].world_position = mesh->world_positions[mesh->indices[i * 3]];
        clipped_vertices[1].world_position = mesh->world_positions[mesh->indices[i * 3 + 1]];
        clipped_vertices[2].world_position = mesh->world_positions[mesh->indices[i * 3 + 2]];
        //read normals
        clipped_vertices[0].normal = mesh->world_normals[mesh->indices[i * 3]];
        clipped_vertices[1].normal = mesh->world_normals[mesh->indices[i * 3 + 1]];
        clipped_vertices[2].normal = mesh->world_normals[mesh->indices[i * 3 + 2]];
        //read texture coordinates
        clipped_vertices[0].texcoord = mesh->textureCoords[mesh->texcoordIndices[i * 3]];
        clipped_vertices[1].texcoord = mesh->textureCoords[mesh->texcoordIndices[i * 3 + 1]];
        clipped_vertices[2].texcoord = mesh->textureCoords[mesh->texcoordIndices[i * 3 + 2]];

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
            // back-face =  ( v0 - viewer position) dot N >= 0 ,  it is not recommended because this rasterization has automatic back-face culling!
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
            uint32_t x0 = max(int32_t(0), (int32_t)(std::floor(xmin)));
            uint32_t x1 = min(int32_t(ImageWidth) - 1, (int32_t)(std::floor(xmax)));
            //uint32_t dx1x0 = x1 - x0; // BBox width
            uint32_t y0 = max(int32_t(0), (int32_t)(std::floor(ymin)));
            uint32_t y1 = min(int32_t(ImageHeight) - 1, (int32_t)(std::floor(ymax)));
            //uint32_t dy1y0 = y1 - y0; // BBox height

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
            XMVECTOR PxPyPz = XMVectorSet(x0 + 0.5f, y0 + 0.5f, 0.0f, 0.0f);
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
            for (uint32_t y = y0; y <= y1; y++)
            {
                //float w0 = w_0 , w1 = w_1 , w2 = w_2;
                XMVECTOR w0_w1_w2 = w_0_w_1_w_2;
                for (uint32_t x = x0; x <= x1; x++)
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
                    if ((mask & 7) == 0)
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
                            //float RdotV = max(0.0f, XMVectorGetX(XMVector3Dot(R, V)));// phong brdf

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
                              //  frameBuffer[(uint64_t)y * ImageWidth + x] = BGRA(255,255,255,255);

                        }

                    }
                    //edge equation: E(X, Y) = (x - X) * dY - (y - Y) * dX   
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
    float3  look(center - eye); // the direction the camera is looking at will become the z-coordinate of the camera
    look = normalize(look);
    float3 right(cross(look, updirection));// the x-coordinate of camera
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
