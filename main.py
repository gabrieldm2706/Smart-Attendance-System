from fastapi import FastAPI
from app.routes.students import router as students_router
from app.database import get_connection

from app.routes.sections import router as sections_router
from app.routes.attendance import router as attendance_router
from app.routes.faces import router as faces_router
from app.routes.auth_routes import router as auth_router
from app.routes.schedules import router as schedules_router
from app.routes.courses import router as courses_router
from app.routes.kiosk import router as kiosk_router, lifespan as kiosk_lifespan

from pathlib import Path
from fastapi.responses import FileResponse

from fastapi import Request
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="Smart Attendance System",
    description="Machine Learning-Based Smart Attendance System",
    version="1.0.0",
    lifespan=kiosk_lifespan,
)

app.include_router(students_router)
app.include_router(faces_router)
app.include_router(sections_router)
app.include_router(attendance_router)
app.include_router(auth_router)
app.include_router(schedules_router)
app.include_router(courses_router)
app.include_router(kiosk_router)

@app.get("/")
def root():
    return {
        "message": "Smart Attendance System API is running"
    }

@app.get("/login", include_in_schema=False)
def login_page():
    login_file = Path(__file__).resolve().parent / "static" / "login.html"
    return FileResponse(login_file)

@app.get("/dashboard", include_in_schema=False)
def dashboard(request: Request):
    token = request.cookies.get("access_token")

    if not token:
        return RedirectResponse(url="/login", status_code=303)

    try:
        from app.auth import decode_access_token
        decode_access_token(token)
    except Exception:
        return RedirectResponse(url="/login", status_code=303)

    dashboard_file = (
        Path(__file__).resolve().parent
        / "static"
        / "dashboard.html"
    )
    return FileResponse(dashboard_file)


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/database-test")
def database_test():

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT DATABASE()")
        database_name = cursor.fetchone()[0]

        return {
            "status": "success",
            "database": database_name
        }

    except Exception as error:

        return {
            "status": "error",
            "message": str(error)
        }

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


@app.get("/register", include_in_schema=False)
def registration_page():
    page_path = Path(__file__).parent / "static" / "registration.html"
    return FileResponse(page_path)