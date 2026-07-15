const API="/books";

loadBooks();

function loadBooks(){

fetch(API)

.then(r=>r.json())

.then(data=>{

let html="";

data.forEach(book=>{

html+=`

<div class="card">

<h3>${book.title}</h3>

<p><b>Author:</b> ${book.author}</p>

<p><b>Quantity:</b> ${book.quantity}</p>

<div class="actions">

<button class="edit"

onclick="editBook(${book.id},
'${book.title}',
'${book.author}',
${book.quantity})">

Edit

</button>

<button class="delete"

onclick="deleteBook(${book.id})">

Delete

</button>

</div>

</div>

`;

});

document.getElementById("books").innerHTML=html;

});

}

function addBook(){

const title=document.getElementById("title").value;

const author=document.getElementById("author").value;

const quantity=document.getElementById("quantity").value;

fetch(API,{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

title,

author,

quantity

})

})

.then(()=>{

loadBooks();

document.getElementById("title").value="";

document.getElementById("author").value="";

document.getElementById("quantity").value="";

});

}

function deleteBook(id){

fetch(API+"/"+id,{

method:"DELETE"

})

.then(()=>loadBooks());

}

function editBook(id,title,author,quantity){

let t=prompt("Title",title);

let a=prompt("Author",author);

let q=prompt("Quantity",quantity);

if(t==null) return;

fetch(API+"/"+id,{

method:"PUT",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

title:t,

author:a,

quantity:q

})

})

.then(()=>loadBooks());

}