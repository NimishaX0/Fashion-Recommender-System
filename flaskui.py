from flask import Flask, request, render_template
from collabFiltering import recommend_products, model, num_articles, newUser_Recommend,newUserDate  # Make sure these are imported

app = Flask(__name__)

#@app.route("/")
#def hello_world():
#    return "<p>HELLO WORLD</p>"

#@app.route('/home', methods=['GET', 'POST'])
#def home():
 #   recommendations = None
  #  if request.method == 'POST':
   #     user_id = int(request.form['user_id'])
    #    recommendations = recommend_products(model, user_id, num_articles , top_n=5)
    # return render_template('index.html', recommendations=recommendations)


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        is_new_user = request.form.get("is_new_user")
        user_id = request.form.get("user_id")

        if is_new_user:

            recommendations = newUser_Recommend(newUserDate)
        else:
            try:
                user_id = int(user_id)

                recommendations = recommend_products(model, user_id, num_articles,5)
            except ValueError:
                recommendations = ["Invalid user ID."]

    return render_template("index.html", recs=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

