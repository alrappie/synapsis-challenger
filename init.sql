-- Connect to the database
\c synapsis;

-- Create the person_tracking table if it doesn't exist
CREATE TABLE IF NOT EXISTS person_tracking (
    id SERIAL PRIMARY KEY,
    person_id INT,
    event TEXT,
    x FLOAT,    
    y FLOAT,
    area_name TEXT,
    polygon_coordinates TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);