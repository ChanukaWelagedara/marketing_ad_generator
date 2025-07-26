from generate_ad_text import generate_ad
from generate_image_from_ad import generate_image_from_text

prompt = """Write a romantic Facebook ad targeting 25 to 35-year-old males
looking to surprise their partners on Valentine’s Day.
The product is a 'Valentine’s Day Flower Bouquet' with features:
fresh roses, elegant wrapping, same-day delivery, and handwritten note."""
ad_text = generate_ad(prompt)
print("\n=== Generated Ad ===")
print(ad_text)

generate_image_from_text(ad_text)
