
import sqlite3

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_required, login_user, logout_user, current_user

from twinky.config import ARTISTS
from twinky.config import ART_PATH
from twinky.config import SECRET
from twinky.learning import get_predictions_for_artist

from model.user import User

login_manager = LoginManager()

app = Flask(__name__,
            static_url_path='',
            static_folder='Datasets')
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True
login_manager.init_app(app)
login_manager.login_view = 'login'
app.secret_key = SECRET


@app.route("/")
@login_required
def index():
	return render_template("index.html", artists=ARTISTS)

@app.route("/dashboard")
@login_required
def dashboard():
	return render_template("dashboard.html")


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', 'guest')
        password = request.form.get('password', '')
        user = user_by_username(username, password)
        if user:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Username or Password Error')
            return render_template('login.html')
    return render_template('login.html')


@app.route("/api/predict", methods=["GET"])
def predict():
    artist_name = ARTISTS[int(request.args.get('artist'))]
    results = get_predictions_for_artist(artist_name)
    return render_template("index.html", artists=ARTISTS, result=results, base_path=ART_PATH)


@login_manager.user_loader
def load_user(user_id):
   conn = sqlite3.connect('twinky.db')
   curs = conn.cursor()
   curs.execute("SELECT * from login where user_id = (?)", [user_id])
   db_user = curs.fetchone()
   return User(int(db_user[0]), db_user[1], db_user[2]) if db_user is not None else None


def user_by_username(username, password):
   conn = sqlite3.connect('twinky.db')
   curs = conn.cursor()
   curs.execute("SELECT * from login where username = (?) AND password = (?)", [username, password])
   db_user = curs.fetchone()
   return User(int(db_user[0]), db_user[1], db_user[2]) if db_user is not None else None


print('running dev server')
app.debug = False
app.run('0.0.0.0', port=1080)
