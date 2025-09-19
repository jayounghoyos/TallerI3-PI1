import os
import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Visualize embeddings of a random movie"

    def handle(self, *args, **kwargs):
        # ✅ Get all movies
        movies = Movie.objects.all()
        
        if not movies.exists():
            self.stderr.write("No movies found in database")
            return
        
        # ✅ Select a random movie
        random_movie = random.choice(movies)
        
        self.stdout.write(f"🎬 Selected random movie: {random_movie.title}")
        self.stdout.write(f"📝 Description: {random_movie.description}")
        self.stdout.write(f"🎭 Genre: {random_movie.genre}")
        self.stdout.write(f"📅 Year: {random_movie.year}")
        
        # ✅ Retrieve and display embedding
        try:
            embedding_vector = np.frombuffer(random_movie.emb, dtype=np.float32)
            
            self.stdout.write(f"\n🔢 Embedding Information:")
            self.stdout.write(f"   - Vector dimension: {len(embedding_vector)}")
            self.stdout.write(f"   - Data type: {embedding_vector.dtype}")
            self.stdout.write(f"   - Min value: {np.min(embedding_vector):.6f}")
            self.stdout.write(f"   - Max value: {np.max(embedding_vector):.6f}")
            self.stdout.write(f"   - Mean value: {np.mean(embedding_vector):.6f}")
            
            self.stdout.write(f"\n📊 First 10 embedding values:")
            for i, value in enumerate(embedding_vector[:10]):
                self.stdout.write(f"   [{i:2d}]: {value:.6f}")
            
            self.stdout.write(f"\n📊 Last 10 embedding values:")
            for i, value in enumerate(embedding_vector[-10:], len(embedding_vector)-10):
                self.stdout.write(f"   [{i:2d}]: {value:.6f}")
            
            # ✅ Check if embedding is default (random) or real
            if np.allclose(embedding_vector, np.random.rand(1536), atol=1e-6):
                self.stdout.write(f"\n⚠️  WARNING: This appears to be a default random embedding!")
                self.stdout.write(f"   Run 'python manage.py movie_embeddings' to generate real embeddings.")
            else:
                self.stdout.write(f"\n✅ This appears to be a real embedding from OpenAI!")
                
        except Exception as e:
            self.stderr.write(f"❌ Error reading embedding: {e}")
            self.stdout.write(f"   Make sure the embedding field contains valid binary data.")
        
        self.stdout.write(f"\n🎉 Embedding visualization completed!")
