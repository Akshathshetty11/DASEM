import os
from app import create_app
from app.models import db

app = create_app()

# Automatically ensure database tables exist when starting app
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print(f"🚀 Starting Dynamic Accident Severity Estimation Model Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
