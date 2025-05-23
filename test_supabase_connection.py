
import os
from dotenv import load_dotenv
from supabase import create_client
import requests

load_dotenv()

# Obtener las variables de entorno
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

print(f"URL de Supabase: {supabase_url}")
print(f"API Key de Supabase: {supabase_key[:5]}...{supabase_key[-5:] if supabase_key else ''}")

# Probar conectividad básica (sin client)
try:
    response = requests.get(f"{supabase_url}/rest/v1/?apikey={supabase_key}", timeout=5)
    print(f"Respuesta directa HTTP: {response.status_code}")
    print(f"Contenido: {response.text[:100]}...")
except Exception as e:
    print(f"Error al conectar directamente: {e}")

# Intentar crear el cliente
try:
    supabase = create_client(supabase_url, supabase_key)
    print("Cliente Supabase creado correctamente")
    
    # Intentar una operación simple
    response = supabase.table("transporte_aereo").select("count", count="exact").execute()
    print(f"Respuesta de consulta: {response}")
except Exception as e:
    print(f"Error al crear cliente o consultar: {e}")