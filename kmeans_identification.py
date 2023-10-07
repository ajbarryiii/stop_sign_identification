import cv2
import os
import time
import numpy as np

# Define the new working directory path
new_working_directory = '/Users/ajbarry/Dropbox/Documents/MA y2/Applied CV/HW1'

# Change the current working directory
os.chdir(new_working_directory)

def get_box(img):
    
    dist = 100

    # 1. Apply color filtering to isolate red regions of the image
    lower_red = np.array([0, 0, 65])
    upper_red = np.array([120, 120, 255])
    mask = cv2.inRange(img, lower_red, upper_red)
    filtered = cv2.bitwise_and(img, img, mask=mask)
  
    # 2. Reshape pixels for clustering
    pixels = filtered.reshape((-1, 3))
    pixels = np.float32(pixels)
  
    # 3. Apply K-means clustering to segregate by color
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 4
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  
    # 4. Identify and extract the cluster closest to a predefined red color
    red = np.array([38, 35, 167])
    distances = np.linalg.norm(centers - red, axis=1)
    red_cluster_id = np.argmin(distances)
    red_cluster_mask = labels.flatten() == red_cluster_id
    red_cluster_coords = np.column_stack(np.where(red_cluster_mask.reshape(img.shape[:2])))
  
    # 5. Perform spatial K-means clustering on the identified red cluster
    K = 2
    _, loc_labels, loc_centers = cv2.kmeans(np.float32(red_cluster_coords), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  
    # 6. Determine the larger of the two spatial clusters based on the count of points and distance between clusters
    counts = [np.sum(loc_labels == i) for i in range(2)]
    spatial_distance = np.linalg.norm(loc_centers[0] - loc_centers[1])
    larger_cluster_idx = np.argmax(counts)

    # 7. Filter the coordinates based on specific criteria
    if spatial_distance > dist and counts[larger_cluster_idx] >= 2 * counts[1 - larger_cluster_idx]:
        coords_of_interest = red_cluster_coords[loc_labels.flatten() == larger_cluster_idx]
    else:
        coords_of_interest = red_cluster_coords
  
    # 8. Iteratively refine and filter the cluster to remove outliers based on distance from spatial center
    while True:
        spatial_center_x = np.mean(coords_of_interest[:, 1]).astype(int)
        spatial_center_y = np.mean(coords_of_interest[:, 0]).astype(int)
        
        distances_from_center = np.sqrt((coords_of_interest[:, 0] - spatial_center_y)**2 + (coords_of_interest[:, 1] - spatial_center_x)**2)
        distance_high_percentile = np.percentile(distances_from_center, 99.99)
        distance_low_percentile = np.percentile(distances_from_center, 50)
        distance_cutoff = np.percentile(distances_from_center, 97)

        if (distance_high_percentile - distance_low_percentile) > distance_low_percentile:
            coords_of_interest = coords_of_interest[distances_from_center <= distance_cutoff]
        else:
            break
  
    # 9. Compute bounding coordinates for the final cluster
    min_y, min_x = np.min(coords_of_interest, axis=0).astype(int)
    max_y, max_x = np.max(coords_of_interest, axis=0).astype(int)
  
    # 10. Return the bounding coordinates
    return min_x, min_y, max_x, max_y

if __name__ == "__main__":

    start_time = time.time()

    dir_path = './images/'
    for i in range(1, 25):
        img_name = f'stop{i}.png'
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        # Get the coordinators of the box
        xmin, ymin, xmax, ymax = get_box(img)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        output_path = f'./results/{img_name}'
        cv2.imwrite(output_path, img)

    end_time = time.time()
    # Make it < 30s
    print(f"Running time: {end_time - start_time} seconds")


