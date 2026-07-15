from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///library.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Book Model
class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

# Create Database
with app.app_context():
    db.create_all()

# Home Route
@app.route("/")
def home():
    return render_template("index.html")

# Get All Books
@app.route("/books", methods=["GET"])
def get_books():
    books = Book.query.all()

    data = []

    for book in books:
        data.append({
            "id": book.id,
            "title": book.title,
            "author": book.author,
            "quantity": book.quantity
        })

    return jsonify(data)

# Get One Book
@app.route("/books/<int:id>", methods=["GET"])
def get_book(id):
    book = Book.query.get_or_404(id)

    return jsonify({
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "quantity": book.quantity
    })

# Add Book
@app.route("/books", methods=["POST"])
def add_book():
    data = request.json

    new_book = Book(
        title=data["title"],
        author=data["author"],
        quantity=data["quantity"]
    )

    db.session.add(new_book)
    db.session.commit()

    return jsonify({"message": "Book Added Successfully"})

# Update Book
@app.route("/books/<int:id>", methods=["PUT"])
def update_book(id):
    book = Book.query.get_or_404(id)

    data = request.json

    book.title = data["title"]
    book.author = data["author"]
    book.quantity = data["quantity"]

    db.session.commit()

    return jsonify({"message": "Book Updated Successfully"})

# Delete Book
@app.route("/books/<int:id>", methods=["DELETE"])
def delete_book(id):
    book = Book.query.get_or_404(id)

    db.session.delete(book)
    db.session.commit()

    return jsonify({"message": "Book Deleted Successfully"})

if __name__ == "__main__":
    app.run(debug=True)