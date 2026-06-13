# Talent Scout AI

### A talent scount software for HR

#### Setup
1. Create your .env file
<pre>
cp .env_example .env 
</pre>

2. Get your [Google AI API](https://aistudio.google.com/api-keys) and copy it into .env file

3. Create virtualenv:
<pre>
python -m venv venv
</pre>

4. Activate it:
<pre>
#Linux
source venv/bin/activate

#Windows
.\venv\Scripts\activate
</pre>

5. Install requirements.txt:
<pre>
pip install -r requirements.txt
</pre>

#### Execution
<pre>
python main.py
</pre>