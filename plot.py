import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os


def find_first_singing_days(df):
    first_singing_days = []

    for bird in df.columns:
        first_singing_day = df.index[df[bird] > 0][0]
        first_singing_days.append(first_singing_day)  
    
    return first_singing_days


def arrange_data(df):
    # britts = pd.read_csv(path_to_britts, index_col=0)
    bird_names = df.columns.tolist()
    
    first_singing_days = find_first_singing_days(df)
   
    new_df = pd.DataFrame()
    new_df['day_started_singing'] = first_singing_days
    new_df['bird_names'] = bird_names
    new_df = new_df.set_index('bird_names')
    
    return new_df


def plot_days_to_start_singing(path_to_britts, path_to_scotts):
    
    #Arrange data
    britts = pd.read_csv(path_to_britts, index_col=0)
    basic_data_britts = arrange_data(britts)
    
    scotts = pd.read_csv(path_to_scotts, index_col=0)
    basic_data_scotts = arrange_data(scotts)

    #Calculate
    days_from_start_britts = basic_data_britts.groupby('day_started_singing').size().tolist()
    days_from_start_scotts = basic_data_scotts.groupby('day_started_singing').size().tolist()
    days_from_start_scotts.append(0)

    # Plot the data:
    plt.plot(np.arange(4), days_from_start_britts, color='blue', label='Britts, n=12')
    plt.plot(np.arange(4), days_from_start_scotts, color='green', label='Scotts, n=10')
    plt.axvline(np.mean(find_first_singing_days(britts)), color='b', linestyle='--', label='britts mean')
    plt.axvline(np.mean(find_first_singing_days(scotts)), color='g', linestyle='--', label='scotts mean')
    plt.xticks(np.arange(4))
    plt.xlabel('Time to start singing (days)')
    plt.ylabel('Count')
    plt.title(f"Britts vs Scotts\nTime to start singing")
    plt.legend()
    plt.show()


def plot_pie_chart_days_to_start_singing(path_to_britts, path_to_scotts, save=False, format='png'):
    #Arrange data
    britts = pd.read_csv(path_to_britts, index_col=0)
    basic_data_britts = arrange_data(britts)
    
    scotts = pd.read_csv(path_to_scotts, index_col=0)
    basic_data_scotts = arrange_data(scotts)

    #Calculate
    days_from_start_britts = basic_data_britts.groupby('day_started_singing').size().tolist()
    days_from_start_scotts = basic_data_scotts.groupby('day_started_singing').size().tolist()
    days_from_start_scotts.append(0)

    under_over_2_split_britts = [(basic_data_britts['day_started_singing'] <= 1).sum(), (basic_data_britts['day_started_singing'] > 1).sum()]
    under_over_2_split_scotts = [(basic_data_scotts['day_started_singing'] <= 1).sum(), (basic_data_scotts['day_started_singing'] > 1).sum()]

    #plot pie chart showing for each group, what part started singing in each day from start
    labels1 = ['0', '1', '2', '3']
    labels2 = ['0-1', '2-3']
    fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    # Plot the first pie chart
    ax1.pie(days_from_start_britts, labels=labels1, autopct='%1.1f%%', startangle=140)
    ax1.set_title('Britts, n=12')

    # Plot the second pie chart
    ax2.pie(days_from_start_scotts, labels=labels1, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Scotts, n=10')

    # Plot the third pie chart
    ax3.pie(under_over_2_split_britts, labels=labels2, autopct='%1.1f%%', startangle=140)

    # Plot the fourth pie chart
    ax4.pie(under_over_2_split_scotts, labels=labels2, autopct='%1.1f%%', startangle=140)

    # Create a joint legend for the top two subplots
    handles1, _ = ax1.get_legend_handles_labels()
    fig.legend(handles1, labels1, title="Days to Start Singing (Top)", loc="upper center", ncol=len(labels1), bbox_to_anchor=(0.5, 0.55))

    # Create a joint legend for the bottom two subplots
    handles2, _ = ax3.get_legend_handles_labels()
    fig.legend(handles2, labels2, title="Days to Start Singing (Bottom)", loc="lower center", ncol=len(labels2), bbox_to_anchor=(0.5, 0.05))

    # handles, _ = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, title="Days To Start Singing", loc="lower center", ncol=4)

    fig.suptitle("Time-to-start-singing splits in each group (in days)")
    # Adjust layout
    plt.tight_layout()
    if save==True:
        basename = f"time_to_start_singing.{format}"
        filename = os.path.join('/Users/cohenlab/Desktop/battle_of_the_islands/figures', basename)
        plt.savefig(filename, format=format)

    plt.show()



def plot_mean_songs_per_day(path_to_britts, path_to_scotts, errorbars=True, save=False, format='png'):

    britts = pd.read_csv(path_to_britts, index_col=0)
    britts.replace(0, np.nan, inplace=True)
    print(britts)

    scotts = pd.read_csv(path_to_scotts, index_col=0)
    scotts.replace(0, np.nan, inplace=True)
    print(scotts)


    # Calculate mean songs per day for each group
    mean_songs_Britts = britts.mean(axis=1)
    mean_songs_Scotts = scotts.mean(axis=1)

    error_Britts = britts.std(axis=1)
    error_Scotts = scotts.std(axis=1)

    # Plotting
    if errorbars:
        plt.plot(britts.index, mean_songs_Britts, color='blue')
        plt.plot(scotts.index, mean_songs_Scotts, color='green')
        plt.errorbar(britts.index, mean_songs_Britts, yerr=error_Britts, label='Britts, n=12', fmt='o-', capsize=5, color='blue', alpha=0.8)
        plt.errorbar(scotts.index, mean_songs_Scotts, yerr=error_Scotts, label='Scotts, n=10', fmt='o-', capsize=5, color='green', alpha=0.8)
    else:
        plt.plot(britts.index, mean_songs_Britts, color='blue', label='Britts, n=12')
        plt.plot(scotts.index, mean_songs_Scotts, color='green', label='Scotts, n=10')

    # Add labels and title
    plt.xlabel('Day')
    plt.ylabel('Mean Songs Sang')
    plt.title('Mean Songs Sang per Day Britts vs. Scotts')
    plt.legend()

    # Show plot
    # plt.grid(True)
    plt.tight_layout()
    if save==True:
        basename = f"mean_songs_sang_per_day.{format}"
        filename = os.path.join('/Users/cohenlab/Desktop/battle_of_the_islands/figures', basename)
        plt.savefig(filename, format=format)

    plt.show()


if __name__ == "__main__": 

    path_to_britts = r'/Users/cohenlab/Desktop/battle_of_the_islands/songs_per_day/britts_songs_per_day.csv'
    path_to_scotts = r'/Users/cohenlab/Desktop/battle_of_the_islands/songs_per_day/scotts_songs_per_day.csv'


    # plot_mean_songs_per_day(path_to_britts, path_to_scotts, save=True, format='svg')
  
    # plot_days_to_start_singing(path_to_britts, path_to_scotts)

    plot_pie_chart_days_to_start_singing(path_to_britts, path_to_scotts, save=True, format='svg')
