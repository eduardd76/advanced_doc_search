#!/usr/bin/env python3
"""
Startup script for Advanced Document Search System
"""

import sys
import subprocess
import os
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import uvicorn
        import fastapi
        import sentence_transformers
        print("✅ Python dependencies installed")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check if Node.js is available for frontend
    try:
        subprocess.run(['node', '--version'], capture_output=True, check=True)
        print("✅ Node.js is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Node.js not found. Frontend will not be available.")
        print("Install Node.js from https://nodejs.org/")
    
    return True

def download_models():
    """Download required models if not already cached"""
    print("📥 Ensuring models are downloaded...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download embedding model
        print("   Checking sentence transformer...")
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            del model  # Free memory
            print("   ✓ Sentence transformer ready")
        except Exception as e:
            print(f"   ⚠️ Sentence transformer will be downloaded on first use: {e}")
        
        # Cross-encoder will be downloaded when needed
        print("   ✓ Cross-encoder will be downloaded on first use")
        
        print("✅ Models configured")
        
    except Exception as e:
        print(f"⚠️  Model setup warning: {e}")
        print("   Models will be downloaded on first use.")

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    
    backend_dir = Path(__file__).parent / "backend"
    
    if not backend_dir.exists():
        print(f"❌ Backend directory not found at: {backend_dir}")
        return None
    
    # Start backend in background - using simple version to avoid memory issues
    cmd = [sys.executable, str(backend_dir / "main_simple.py")]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=str(backend_dir)
    )
    
    # Wait for server to start
    print("   Waiting for backend to start...")
    time.sleep(5)
    
    # Check if process is still running
    if process.poll() is None:
        print("✅ Backend server started at http://localhost:8000")
        return process
    else:
        stdout, stderr = process.communicate()
        print(f"❌ Backend failed to start:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return None

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting frontend...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    if not (frontend_dir / "node_modules").exists():
        print("   Installing frontend dependencies...")
        result = subprocess.run(
            ['npm', 'install'], 
            capture_output=True,
            cwd=str(frontend_dir),
            shell=True  # Use shell on Windows
        )
        if result.returncode != 0:
            print("❌ Failed to install frontend dependencies")
            print(f"   Error: {result.stderr.decode('utf-8', errors='ignore')}")
            return None
    
    # Start frontend
    print("   Starting React development server...")
    process = subprocess.Popen(
        'npm start',  # Use string command on Windows
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=str(frontend_dir),
        shell=True  # Use shell on Windows
    )
    
    # Wait for frontend to start
    time.sleep(10)
    
    if process.poll() is None:
        print("✅ Frontend started at http://localhost:3003")
        return process
    else:
        print("❌ Frontend failed to start")
        return None

def main():
    """Main startup function"""
    print("🔥 Advanced Document Search System")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        # Don't change directory, work with absolute paths
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Download models
        download_models()
        
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            return 1
        
        # Start frontend
        frontend_process = start_frontend()
        
        print("\n" + "=" * 50)
        print("🎉 System started successfully!")
        print("📚 Backend API: http://localhost:8000")
        print("🌐 Web Interface: http://localhost:3003")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("=" * 50)
        
        # Open browser
        try:
            webbrowser.open('http://localhost:3003')
        except Exception:
            pass
        
        print("\n💡 Tips:")
        print("   • Create an index from the 'Index' tab")
        print("   • Search your documents from the 'Search' tab")
        print("   • Evaluate system performance from the 'Evaluate' tab")
        print("   • Chat with your documents from the 'Chat' tab")
        print("\n⚠️  Press Ctrl+C to stop all servers")
        
        # Keep processes running
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if backend_process.poll() is not None:
                    print("❌ Backend process stopped")
                    break
                    
                if frontend_process and frontend_process.poll() is not None:
                    print("❌ Frontend process stopped")
                    # Continue running with just backend
                    frontend_process = None
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            
            # Terminate processes
            if backend_process:
                backend_process.terminate()
                backend_process.wait()
                print("✅ Backend stopped")
                
            if frontend_process:
                frontend_process.terminate()
                frontend_process.wait()
                print("✅ Frontend stopped")
        
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
        
    finally:
        pass  # No need to change back directory

if __name__ == '__main__':
    sys.exit(main())