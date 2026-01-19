#include "pch.h"
#include "Render.h"

using namespace winrt;

using namespace Windows;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::Foundation::Numerics;
using namespace Windows::UI;
using namespace Windows::UI::Core;
using namespace Windows::UI::Composition;
using namespace Windows::Graphics::Display;
using namespace Windows::System;
using namespace Windows::UI::ViewManagement;

uint32_t ImageWidth, ImageHeight;
constexpr bool useOrbitalCamera = false; // switch between Orbital and Fps camera 

struct App : implements<App, IFrameworkViewSource, IFrameworkView>
{
   
    IFrameworkView CreateView()
    {
        return *this;
    }

    void Initialize(CoreApplicationView const & appView)
    {
        appView.Activated({ this,&App::OnActivated });              
    }

    void Load(hstring const&)
    {
        m_Render = make_unique<Render>();        
    }

    void Uninitialize()
    {
       
    }

    void Run()
    {
        com_ptr<ID3D11Device> device;
        com_ptr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,D3D_FEATURE_LEVEL_9_3,D3D_FEATURE_LEVEL_9_2,
            D3D_FEATURE_LEVEL_9_1
        }, featurelevel{};

        UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
        D2D1_FACTORY_OPTIONS factoryOptions = { D2D1_DEBUG_LEVEL_INFORMATION };
        com_ptr<ID2D1Factory> factory;
        check_hresult(
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, factoryOptions, factory.put())
        );
        m_d2dFactory = factory.as<ID2D1Factory1>();
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
        check_hresult(
            D3D11CreateDevice(
                nullptr,
                D3D_DRIVER_TYPE_HARDWARE,
                0,
                creationFlags, featureLevels,
                _countof(featureLevels),
                D3D11_SDK_VERSION,
                device.put(),
                &featurelevel,
                context.put())
        );
#else
        com_ptr<ID2D1Factory> factory;
        check_hresult(
            D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, factory.put())
        );
        m_d2dFactory = factory.as<ID2D1Factory1>();
        check_hresult(
            D3D11CreateDevice(
                nullptr,
                D3D_DRIVER_TYPE_HARDWARE,
                0,
                creationFlags, featureLevels,
                _countof(featureLevels),
                D3D11_SDK_VERSION,
                device.put(),
                &featurelevel,
                context.put())
        );
#endif
        
       
        
        m_d3dDevice = device.as<ID3D11Device1>();
        m_d3dContext = context.as<ID3D11DeviceContext1>();
        m_dxgiDevice = device.as<IDXGIDevice1>();
        check_hresult(
            m_d2dFactory->CreateDevice(m_dxgiDevice.get(), m_d2dDevice.put())
        );
        check_hresult(
            m_d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, m_d2dContext.put())
        );
        check_hresult(
            DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED,
                __uuidof(IDWriteFactory),
                reinterpret_cast<IUnknown**>(put_abi(m_dwriteFactory))     //same as  reinterpret_cast<IUnknown**>(m_dwriteFactory.put())
            )
        );
        check_hresult(
            m_dwriteFactory->CreateTextFormat(
                L"Consolas",
                nullptr,
                DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                30.0f,
                L"",
                m_dwriteTextFormat.put())
        );

        CreateWindowSizeDependentResources();
       
       
        while (!m_windowClosed)
        {
            CoreDispatcher dispatcher = m_window.get().Dispatcher();
            dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);

            ID3D11RenderTargetView* targets[1] = { m_renderTarget.get() };
            m_d3dContext->OMSetRenderTargets(1, targets, nullptr);
            const float clearColor[4] = { 0.0f,0.5f,1.0f,1.0f };
            m_d3dContext->ClearRenderTargetView(m_renderTarget.get(), clearColor);
            D2D1_SIZE_F sizeRenderTarget = m_d2dContext->GetSize();
            com_ptr<ID2D1SolidColorBrush> m_brush;
            check_hresult(
                m_d2dContext->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Coral), m_brush.put())
            );
           
            if (useOrbitalCamera)
                UpdateOrbitalCamera();
            else
            {               
               m_Render->flyCamera.Update();  
               m_Render->flyCamera.SetPerspective(m_Render->flyCamera.GetFov(), (float)ImageWidth / ImageHeight, nearClippingPLane, farClippingPLane);         
            }
            
          
            double ms=0.0;
           
            m_Render->RenderMain(&ms);
            renderTime = L"Render time :  ";
            renderTime += std::to_wstring(ms); //1 frame rendered ms milliseconds
            renderTime += L" ms , " + std::to_wstring(1000.0f/ms) + L" fps";
                      
            m_bitmapRasterized.detach();            
             check_hresult(
                 m_d2dContext->CreateBitmap(
                     D2D1::SizeU(ImageWidth, ImageHeight),
                     m_Render->p_bgra,
                     ImageWidth * 4,
                     D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_NONE, D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE)),
                     m_bitmapRasterized.put())
             );

            m_d2dContext->BeginDraw();
            //render code place here....
            m_d2dContext->DrawBitmap(m_bitmapRasterized.get());
            // display the elapsed time of rendering
            m_d2dContext->DrawText(renderTime.c_str(),
                renderTime.size(),
                m_dwriteTextFormat.get(),
                D2D1::RectF(0, 0, sizeRenderTarget.width, sizeRenderTarget.height),
                m_brush.get());

            m_d2dContext->EndDraw();
            check_hresult(
                m_swapchain->Present(1, 0)
            );
            
            m_bitmapRasterized->Release();
        }
    }

    void SetWindow(CoreWindow const & window)
    {
        window.KeyDown({ this,&App::OnKeyDown });
        window.PointerPressed({ this, &App::OnPointerPressed });
        window.PointerMoved({ this, &App::OnPointerMoved });
        window.SizeChanged([&](auto && ...)
            {
                if (m_renderTarget)
                {
                    m_renderTarget = nullptr;               
                    m_bitmapTarget = nullptr;                   
                    m_d2dContext->SetTarget(nullptr);
                    
                    ImageWidth = (uint32_t)m_window.get().GetForCurrentThread().Bounds().Width * dpi / 96.f;
                    ImageHeight = (uint32_t)m_window.get().GetForCurrentThread().Bounds().Height * dpi / 96.f;
                    CreateWindowSizeDependentResources();          

                   
                    // render the image
                   
                    double ms;
                  
                    m_Render->RenderMain(&ms);
                   
                    renderTime = L"Render time :  ";
                    renderTime += std::to_wstring(ms);
                    renderTime += L" ms , " + std::to_wstring(1000.0f / ms) + L" fps";

                    // create a bitmap from rendered image
                    check_hresult(
                        m_d2dContext->CreateBitmap(
                            D2D1::SizeU(ImageWidth, ImageHeight),
                            m_Render->p_bgra,
                            ImageWidth * 4,
                            D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_NONE, D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE)),
                            m_bitmapRasterized.put()));
                }
            });
        window.Closed({ this,&App::OnClosed });
       
    }
    void OnKeyDown(IInspectable const& sender, KeyEventArgs const& args)
    {
       
        if (!useOrbitalCamera) {
            switch (args.VirtualKey())
            {
            case VirtualKey::W:
                m_Render->flyCamera.Walk(-0.80f); 
                break;
            case VirtualKey::S:
                m_Render->flyCamera.Walk(0.80f);
                break;
            case VirtualKey::A:
                m_Render->flyCamera.Strafe(-0.80f);
                break;
            case VirtualKey::D:
                m_Render->flyCamera.Strafe(0.80f);
            }
            m_Render->flyCamera.Update();
            
        }
    }

    void OnPointerPressed(IInspectable const &, PointerEventArgs const & args)
    {
        GetPointerLastPosition().x = args.CurrentPoint().Position().X * GetDpi() / 96.0f;
        GetPointerLastPosition().y = args.CurrentPoint().Position().Y * GetDpi() / 96.0f;
    }

    void OnPointerMoved(IInspectable const &, PointerEventArgs const & args)
    {
       float x = args.CurrentPoint().Position().X * GetDpi() / 96.0f;
       float y = args.CurrentPoint().Position().Y * GetDpi() / 96.0f;
       float LastPositionX = GetPointerLastPosition().x;
       float LastPositionY = GetPointerLastPosition().y;
       float dx, dy;
       if (useOrbitalCamera)
       {
           if (args.CurrentPoint().Properties().IsLeftButtonPressed())
           {
               dx = XMConvertToRadians(0.05f * (x - LastPositionX));
               dy = XMConvertToRadians(0.05f * (y - LastPositionY));
               m_CameraAngleTheta += dx;
               m_CameraAnglePhi = clamp(m_CameraAnglePhi + dy, 0.1f, XM_PI - 0.1f);
           }
           else if (args.CurrentPoint().Properties().IsRightButtonPressed())
           {
               dx = 0.15f * (x - LastPositionX);
               dy = 0.15f * (y - LastPositionY);
               m_CameraOrbitRadius = clamp(m_CameraOrbitRadius + (dx - dy), 20.0f, 200.0f);
           }
       }
       else
       {
           if (args.CurrentPoint().Properties().IsLeftButtonPressed())
           {
               dx = XMConvertToRadians(0.25f * (x - LastPositionX));
               dy = XMConvertToRadians(0.25f * (y - LastPositionY));
               m_Render->flyCamera.RotateUpDown(dy);
               m_Render->flyCamera.RotateLeftRight(dx);
               
           }
       }
      
       GetPointerLastPosition().x = x;
       GetPointerLastPosition().y = y;
    }

    void OnActivated(CoreApplicationView const& appView, IActivatedEventArgs const& args)
    {
        m_window = CoreWindow::GetForCurrentThread();
        m_window.get().Activate();
      
        dpi = DisplayInformation::GetForCurrentView().LogicalDpi(); 
        ImageWidth = (uint32_t) m_window.get().GetForCurrentThread().Bounds().Width * dpi/96.f ;
        ImageHeight = (uint32_t)m_window.get().GetForCurrentThread().Bounds().Height * dpi/96.f;
    }

    void OnClosed(IInspectable const&, CoreWindowEventArgs const& args)
    {
        m_windowClosed = true;
    }
    float2& GetPointerLastPosition() { return m_PointerLastPosition; }

    float GetDpi() { return dpi; }

    void UpdateOrbitalCamera()
    {
        float x = m_CameraOrbitRadius * cosf(m_CameraAngleTheta) * sinf(m_CameraAnglePhi);
        float y = m_CameraOrbitRadius * cosf(m_CameraAnglePhi);
        float z = m_CameraOrbitRadius * sinf(m_CameraAngleTheta) * sinf(m_CameraAnglePhi);
        
       
        m_Render->flyCamera.SetLookAt(float3(x,y,z), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
        m_Render->flyCamera.SetPerspective(m_Render->flyCamera.GetFov(), (float)ImageWidth / ImageHeight, nearClippingPLane, farClippingPLane);
    }
    private:
        agile_ref<CoreWindow>  m_window = nullptr;
        com_ptr<IDXGISwapChain1> m_swapchain = nullptr;
        com_ptr<ID3D11Device1> m_d3dDevice = nullptr;
        com_ptr<ID3D11DeviceContext1> m_d3dContext = nullptr;
        com_ptr<ID3D11RenderTargetView> m_renderTarget = nullptr;
        com_ptr<ID2D1Factory1>  m_d2dFactory = nullptr;
        com_ptr<ID2D1Device> m_d2dDevice = nullptr;
        com_ptr<ID2D1DeviceContext> m_d2dContext = nullptr;
        com_ptr<IDXGIDevice1> m_dxgiDevice = nullptr;
        com_ptr<ID2D1Bitmap1> m_bitmapTarget = nullptr, m_bitmapRasterized = nullptr;
        com_ptr<IDWriteFactory> m_dwriteFactory = nullptr;
        com_ptr<IDWriteTextFormat> m_dwriteTextFormat = nullptr;
        unique_ptr<Render> m_Render{};
        bool m_windowClosed = false;
        std::wstring renderTime;
        float dpi{};
        float m_CameraAngleTheta = 1.5f * XM_PI,
            m_CameraAnglePhi = XM_PIDIV2, m_CameraOrbitRadius = 30.0f;
        float2 m_PointerLastPosition{};
        

        void CreateWindowSizeDependentResources()
        {
            if (m_swapchain != nullptr)
            {
                check_hresult(
                    m_swapchain->ResizeBuffers(2, ImageWidth, ImageHeight, DXGI_FORMAT_B8G8R8A8_UNORM, 0)
                );
            }
            else
            {
                DXGI_SWAP_CHAIN_DESC1 swapChainDesc{ 0 };

                swapChainDesc.Width = ImageWidth;        
                swapChainDesc.Height = ImageHeight;
                swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;          // 24-bit BGRA format
                swapChainDesc.Stereo = false;
                swapChainDesc.SampleDesc.Count = 1;     //no MSAA
                swapChainDesc.SampleDesc.Quality = 0;
                swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
                swapChainDesc.BufferCount = 2;      //double-buffering
                swapChainDesc.Scaling = DXGI_SCALING_NONE;
                swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
                swapChainDesc.Flags = 0;
                check_hresult(
                    m_dxgiDevice->SetMaximumFrameLatency(1)
                );

                com_ptr<IDXGIAdapter> dxgiAdapter;
                check_hresult(
                    m_dxgiDevice->GetAdapter(dxgiAdapter.put())
                );
                com_ptr<IDXGIFactory2> dxgiFactory;
                check_hresult(
                    dxgiAdapter->GetParent(IID_PPV_ARGS(dxgiFactory.put()))
                );
                check_hresult(
                    dxgiFactory->CreateSwapChainForCoreWindow(
                        m_d3dDevice.get(),
                        static_cast<IUnknown*>(get_abi(m_window.get())), // same as get_unknown()
                        &swapChainDesc,
                        nullptr,
                        m_swapchain.put()
                    )
                );
            }
            com_ptr<ID3D11Texture2D> backBuffer;
            check_hresult(
                m_swapchain->GetBuffer(0, IID_PPV_ARGS(backBuffer.put()))
            );
            check_hresult(
                m_d3dDevice->CreateRenderTargetView(backBuffer.get(), nullptr, m_renderTarget.put())
            );
            D3D11_TEXTURE2D_DESC backBufferDesc{ 0 };
            backBuffer->GetDesc(&backBufferDesc);
            D3D11_VIEWPORT viewport{ 0 };
            viewport.TopLeftX = 0.0f;
            viewport.TopLeftY = 0.0f;
            viewport.Width = static_cast<float>(backBufferDesc.Width);
            viewport.Height = static_cast<float>(backBufferDesc.Height);
            viewport.MinDepth = D3D11_MIN_DEPTH;
            viewport.MaxDepth = D3D11_MAX_DEPTH;
            m_d3dContext->RSSetViewports(1, &viewport);

            D2D1_BITMAP_PROPERTIES1 bitmapProp = D2D1::BitmapProperties1(
                D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE),
                0, 0);
            com_ptr<IDXGISurface> dxgiSurface;  // Direct2D need a surface from the back-buffer
            check_hresult(
                m_swapchain->GetBuffer(0, IID_PPV_ARGS(dxgiSurface.put()))
            );
            //create a Bitmap to use as a render target
            check_hresult(
                m_d2dContext->CreateBitmapFromDxgiSurface(
                    dxgiSurface.get(),
                    bitmapProp,
                    m_bitmapTarget.put()
                )
            );
            //set a new render target for 2D display
            m_d2dContext->SetTarget(m_bitmapTarget.get());
        }
    
};

 int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    CoreApplication::Run(make<App>());
}
