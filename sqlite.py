import sqlite3
##connect to database
connection=sqlite3.connect('student.db')

#create a cusrsor object to insert record and create table
cursor=connection.cursor()
##create table

table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

## Insert some records
cursor.execute("insert into STUDENT values('Satya','Data Science','A',100)")
cursor.execute("insert into STUDENT values('Nandish','Web Dev','B',92)")
cursor.execute("insert into STUDENT values('Pankaj','Data Science','A',85)")
cursor.execute("insert into STUDENT values('Hitesh','Dev Ops','B',70)")
cursor.execute("insert into STUDENT values('Rohit','Web Dev','C',78)")

## Display all the records
print("The inserted records are ")
data=cursor.execute("select * from STUDENT")
for row in data:
    print(row)
    
## Commit your changes in the database
connection.commit()
connection.close()