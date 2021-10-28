---
layout: post
title: Post 3
---

# Blog post 3  
## Message Bank   
    Some code/resourse from Professor or TA's code, and this website is build with Flask

### Main page and base page  
These two pages are kind of simple, the main page is basically a welcome and the base page is used as an navigate.

And in app.py we have the following code to render our main page:


```python
from flask import Flask, g, render_template, request
import sqlite3

app = Flask(__name__)

@app.route('/')

def main():
    return render_template('main_better.html')
```

And to run this website we need to open this repository in cmd/terminal to run the following code:    
     export FLASK_ENV=development; flask run (Mac)    
     set FLASK_ENV=development; flask run (Win)

### Submit page    
And now we need to write our page for users to submit their messages.And we could use form tag in html to transmit data.   
The submit page should include codes as follows:


```python
<form method="post" enctype="multipart/form-data" class="cr">
    <label for="message">Your message</label><br>
    <input type="text" name="message" id="message"><br>
    <label for="name">Your name or handle: </label><br>
    <input type="text" name="handle" id="handle"><br>
    <input type="submit" value="Submit"><br>
</form>
```

And we need to add navigate bar to this page, so add the following code to top of the submit.html


```python
{% extends 'base.html' %}
```

And for showing success/error to users:


```python
  {% if thanks %}
    <br>
    <div class="cr">Thanks for submitting</div>
  {% endif %}

  {% if error %}
    <br>
    Oh no, we couldn't read that! 
  {% endif %}
```

Now we need to add code to app.py to get and store the data from the web page.  

   
   First we need a function to connect to the database:      
      The following function will check if the database is in g and connect it if not.  
      Also it will create the table for storing messages from users and create such a table if not exist.


```python
def get_message_db():
    if 'message_db' not in g:
        g.message_db = sqlite3.connect("messages_db.sqlite")

    cursor = g.message_db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS messages(id INT,handle TEXT,message TEXT);")
    
    return g.message_db
```

And we need to get data from users' input and store it into our database.


```python
def insert_message(request):
    message=request.form["message"]
    handle=request.form["handle"]
    #Get data from submit page
    
    db=get_message_db()
    #Connect to database
    
    cursor=db.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages;")
    number_row=cursor.fetchone()[0]
    cursor.execute(f"""INSERT INTO messages (id,handle,message) VALUES ({number_row+1}, "{handle}", "{message}");""")
    db.commit()
    # Add a new row with handle and message and assign a unique id (here is number of row)
    db.close()
```

And write a function to render_template() the submit.html


```python
@app.route('/submit-basic/', methods=['POST', 'GET'])
def submit_basic():
    if request.method == 'GET':
        return render_template('submit-basic.html')
    else:
        try:
            insert_message(request)
            return render_template('submit-basic.html', thanks = True)
        except:
            return render_template('submit-basic.html', error=True)
```

### View page    
This page will pick random message and handle from our database and show it to users.    

First, we need a function that return such messages.


```python
def random_messages(n):
    db=get_message_db()
    cursor=db.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages;")
    number_row=cursor.fetchone()[0]
    if n > number_row:
        n=number_row
    # we don't want the number of messages showed > total messages
    
    a=[]
    for i in range(n):
        cursor.execute("SELECT handle,message FROM messages ORDER BY RANDOM() LIMIT 1;")
        message=cursor.fetchone()
        a.append(message)
    #Pick random data from our database and return such a list
    
    return a
```

And we need to use a loop to showing the messages in view.html


```python
<ul class="cr">
    {% for message in messages %}
      <div class="cr"><li>{{ message[1]|e }}</li>
      <i>--{{ message[0]|e }}</i></div>
    {% endfor %}
</ul>
```

Finally, we write a function to render view.html in app.py


```python
@app.route('/view/')
def view():
    message=random_messages(5)
    #Here we use 5 random messages
    return render_template('view.html',messages=message )
```

Now our website should work properly.   
The final step is to add some "fancy" CSS style  

And this site's repository:
https://github.com/libbyrencc/post/tree/main/post3/website

## Final Screencut

![png](1.png)

![png](2.png)
